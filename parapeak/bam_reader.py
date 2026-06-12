"""
BAM/SAM reading utilities: pileup construction and fragment-size estimation.

Single-end vs paired-end handling
-----------------------------------
Single-end
  Each read is extended from its 5' end by *fragment_size* bp toward the
  3' direction.  fragment_size should be set to the expected sonication /
  tagmentation fragment length (estimated automatically from paired-end
  data if available; default 200 bp).

Paired-end (proper pairs)
  Only R1 is processed; R2 is skipped to avoid double-counting.
  The fragment interval is derived from the SAM TLEN field (template
  length), which encodes the distance from the leftmost to the rightmost
  mapped base of the pair.  This gives the *actual* insert size for each
  fragment rather than a fixed estimate, which is important for:
    - ATAC-seq: the fragment between two Tn5 cut sites IS the accessible
      region; using actual insert sizes reflects nucleosome positioning.
    - CIRCLE-seq / GUIDE-seq: read pairs span the actual cut-site circle;
      actual insert size reflects the circle diameter.
    - Paired-end ChIP-seq: sonication fragments vary; using TLEN avoids
      over-smoothing short fragments or under-covering long ones.

  If TLEN is absent or outside [min_fragment, max_fragment] (e.g. for
  discordant pairs), the read is treated as single-end and extended by
  *fragment_size* from its 5' end.

Paired-end (unpaired / discordant)
  Treated as single-end: extended by *fragment_size* from the 5' end.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pysam

logger = logging.getLogger(__name__)

# Hard caps on insert size accepted from TLEN for paired-end fragments.
_MIN_FRAGMENT = 10    # bp  – smaller inserts are likely alignment artefacts
_MAX_FRAGMENT = 2000  # bp  – larger inserts are likely chimeric/discordant


# ---------------------------------------------------------------------------
# Header utilities
# ---------------------------------------------------------------------------

def get_chromosome_sizes(bam_files: List[str]) -> Dict[str, int]:
    """Return chromosome sizes inferred from BAM/SAM headers (max across files)."""
    sizes: Dict[str, int] = {}
    for path in bam_files:
        mode = 'r' if path.endswith('.sam') else 'rb'
        with pysam.AlignmentFile(path, mode) as bam:
            for ref, length in zip(bam.references, bam.lengths):
                sizes[ref] = max(sizes.get(ref, 0), length)
    return sizes


def estimate_fragment_size(bam_files: List[str], n_reads: int = 100_000) -> int:
    """
    Estimate fragment size from paired-end insert sizes (median of first
    *n_reads* proper pairs).  Falls back to 200 bp for single-end libraries.
    This value is used for single-end extension and as a fallback for
    discordant paired-end reads.
    """
    inserts: List[int] = []
    for path in bam_files:
        if len(inserts) >= n_reads:
            break
        mode = 'r' if path.endswith('.sam') else 'rb'
        with pysam.AlignmentFile(path, mode) as bam:
            for read in bam.fetch():
                if len(inserts) >= n_reads:
                    break
                if (read.is_proper_pair and not read.is_read2
                        and _MIN_FRAGMENT < read.template_length <= _MAX_FRAGMENT):
                    inserts.append(read.template_length)

    if not inserts:
        logger.warning('No paired-end reads found; using default fragment size of 200 bp')
        return 200

    median = int(np.median(inserts))
    logger.info(f'Estimated fragment size: {median} bp (from {len(inserts)} paired reads)')
    return median


def count_mapped_reads(bam_files: List[str]) -> int:
    """
    Count total mapped *fragments*.
    - Paired-end: count only R1 (avoids double-counting).
    - Single-end: count all mapped reads.
    """
    total = 0
    for path in bam_files:
        mode = 'r' if path.endswith('.sam') else 'rb'
        with pysam.AlignmentFile(path, mode) as bam:
            try:
                stats = bam.get_index_statistics()
                mapped = sum(s.mapped for s in stats)
                # Check if library is paired-end from header or first reads
                is_paired = _is_paired_library(bam)
                total += mapped // 2 if is_paired else mapped
            except ValueError:
                total += sum(
                    1 for r in bam.fetch()
                    if not r.is_unmapped and not (r.is_paired and r.is_read2)
                )
    return total


def _is_paired_library(bam: pysam.AlignmentFile, n_check: int = 1000) -> bool:
    """Heuristic: sample first *n_check* reads to detect paired-end library."""
    paired = 0
    for i, read in enumerate(bam.fetch()):
        if i >= n_check:
            break
        if read.is_paired:
            paired += 1
    return paired > n_check // 2


# ---------------------------------------------------------------------------
# Per-chromosome pileup
# ---------------------------------------------------------------------------

def build_pileup_for_chrom(
    bam_files: List[str],
    chrom: str,
    chrom_size: int,
    bin_size: int,
    fragment_size: int,
    blacklist_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build a binned pileup for *chrom* merged across *bam_files*.

    Fragment interval determination per read
    -----------------------------------------
    Paired-end R1, proper pair, TLEN in [_MIN_FRAGMENT, _MAX_FRAGMENT]:
        Use the actual fragment: [reference_start, reference_start + TLEN)
        for forward-oriented R1 (TLEN > 0) or
        [reference_end - |TLEN|, reference_end) for reverse R1 (TLEN < 0).
    Paired-end R2:
        Skipped (fragment already counted via R1).
    All other reads (SE, unpaired PE, discordant PE):
        Extend from 5' end by *fragment_size*:
        forward → [reference_start, reference_start + fragment_size)
        reverse → [reference_end - fragment_size, reference_end)

    Returns
    -------
    pileup    : np.ndarray[float32], shape (n_bins,)
    gc_per_bin: np.ndarray[float64], shape (n_bins,)  – NaN for empty bins
    n_reads   : int  – number of fragments counted
    """
    n_bins = (chrom_size + bin_size - 1) // bin_size
    pileup = np.zeros(n_bins, dtype=np.int32)
    gc_sum = np.zeros(n_bins, dtype=np.float64)
    gc_cnt = np.zeros(n_bins, dtype=np.int32)

    for path in bam_files:
        mode = 'r' if path.endswith('.sam') else 'rb'
        try:
            with pysam.AlignmentFile(path, mode) as bam:
                if chrom not in bam.references:
                    continue
                for read in bam.fetch(chrom, 0, chrom_size):
                    if (read.is_unmapped or read.is_duplicate
                            or read.is_secondary or read.is_supplementary):
                        continue

                    # Skip R2 to avoid double-counting paired fragments
                    if read.is_paired and read.is_read2:
                        continue

                    start, end = _fragment_interval(read, chrom_size, fragment_size)

                    s_bin = start // bin_size
                    e_bin = min(n_bins - 1, (end - 1) // bin_size)
                    if s_bin <= e_bin:
                        pileup[s_bin : e_bin + 1] += 1

                    # Per-bin GC% from read sequence (proxy for reference GC%)
                    seq = read.query_sequence
                    if seq:
                        seq_up = seq.upper()
                        gc = (seq_up.count('G') + seq_up.count('C')) / len(seq_up)
                        b = read.reference_start // bin_size
                        if 0 <= b < n_bins:
                            gc_sum[b] += gc
                            gc_cnt[b] += 1

        except Exception as exc:
            logger.warning(f'Error reading {path} on {chrom}: {exc}')

    if blacklist_mask is not None:
        pileup[blacklist_mask] = 0
        gc_sum[blacklist_mask] = 0
        gc_cnt[blacklist_mask] = 0

    gc_per_bin = np.where(gc_cnt > 0, gc_sum / gc_cnt, np.nan)
    n_reads = int(pileup.sum())
    return pileup.astype(np.float32), gc_per_bin, n_reads


def _fragment_interval(
    read: pysam.AlignedSegment,
    chrom_size: int,
    fragment_size: int,
) -> Tuple[int, int]:
    """
    Compute the [start, end) interval for a single read's fragment.

    Paired-end proper pairs use the actual TLEN-based fragment.
    Everything else uses a fixed extension from the 5' end.
    """
    tlen = read.template_length  # signed; positive on forward R1

    if (read.is_paired and read.is_proper_pair
            and _MIN_FRAGMENT < abs(tlen) <= _MAX_FRAGMENT):
        # Use actual insert size
        if tlen > 0:
            # Forward R1: fragment starts at this read's left end
            start = read.reference_start
            end = min(chrom_size, start + tlen)
        else:
            # Reverse R1 (TLEN < 0 means this read is rightmost)
            ref_end = read.reference_end or (
                read.reference_start + (read.query_length or 1)
            )
            end = ref_end
            start = max(0, end + tlen)  # tlen is negative → subtraction
    else:
        # Single-end or discordant paired-end: fixed extension from 5' end
        if read.is_reverse:
            ref_end = read.reference_end or (
                read.reference_start + (read.query_length or 1)
            )
            end = ref_end
            start = max(0, end - fragment_size)
        else:
            start = read.reference_start
            end = min(chrom_size, start + fragment_size)

    return int(start), int(end)
