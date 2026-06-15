"""
BAM/SAM reading utilities: single-pass pileup construction and fragment-size estimation.

Single-pass design
------------------
build_all_pileups() reads each BAM file exactly once with no region argument,
routing each read to its chromosome buffer.  This is O(N_reads) per file
regardless of sort order, making it suitable for unsorted BAM files.
Sorted + indexed BAM files work identically.

For N chromosomes the old per-chromosome fetch approach required N full scans
of an unsorted file; this implementation requires exactly 1.

Single-end vs paired-end handling
-----------------------------------
Single-end
  Each read is extended from its 5' end by *fragment_size* bp toward the
  3' direction.

Paired-end (proper pairs, TLEN in [min_fragment, max_fragment])
  Only R1 is processed; R2 is skipped to avoid double-counting.
  The fragment interval is derived from the SAM TLEN field (template length).

Paired-end (unpaired / discordant / TLEN out of range)
  Treated as single-end: extended by *fragment_size* from the 5' end.
  These reads are counted in stats['insert_size_fallback'].

QC filters applied per read (in order)
---------------------------------------
1. is_unmapped
2. is_duplicate
3. is_secondary
4. is_supplementary
5. is_qcfail   (SAM flag 0x200)
6. mapping_quality < min_mapq
7. is_read2    (skip to avoid double-counting; not a quality filter)
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pysam

logger = logging.getLogger(__name__)

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


def estimate_fragment_size(
    bam_files: List[str],
    n_reads: int = 100_000,
    min_fragment: int = _MIN_FRAGMENT,
    max_fragment: int = _MAX_FRAGMENT,
) -> int:
    """
    Estimate fragment size from paired-end insert sizes (median of first
    *n_reads* proper pairs within [min_fragment, max_fragment]).
    Falls back to 200 bp for single-end libraries.
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
                        and min_fragment < read.template_length <= max_fragment):
                    inserts.append(read.template_length)

    if not inserts:
        logger.warning('No paired-end reads found; using default fragment size of 200 bp')
        return 200

    median = int(np.median(inserts))
    logger.info(f'Estimated fragment size: {median} bp (from {len(inserts)} paired reads)')
    return median


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
# Fragment interval
# ---------------------------------------------------------------------------

def _fragment_interval(
    read: pysam.AlignedSegment,
    chrom_size: int,
    fragment_size: int,
    min_fragment: int,
    max_fragment: int,
) -> Tuple[int, int, bool]:
    """
    Compute the [start, end) interval for a single read's fragment.

    Returns
    -------
    start, end    : genomic coordinates (0-based, half-open)
    used_fallback : True when a paired read fell back to SE extension
    """
    tlen = read.template_length  # signed; positive on forward R1

    if (read.is_paired and read.is_proper_pair
            and min_fragment < abs(tlen) <= max_fragment):
        if tlen > 0:
            start = read.reference_start
            end = min(chrom_size, start + tlen)
        else:
            ref_end = read.reference_end or (
                read.reference_start + (read.query_length or 1)
            )
            end = ref_end
            start = max(0, end + tlen)
        return int(start), int(end), False

    # Single-end or discordant/out-of-range paired-end: fixed extension from 5' end
    if read.is_reverse:
        ref_end = read.reference_end or (
            read.reference_start + (read.query_length or 1)
        )
        end = ref_end
        start = max(0, end - fragment_size)
    else:
        start = read.reference_start
        end = min(chrom_size, start + fragment_size)

    used_fallback = read.is_paired  # paired but couldn't use TLEN
    return int(start), int(end), used_fallback


# ---------------------------------------------------------------------------
# Single-pass pileup builder
# ---------------------------------------------------------------------------

def _empty_stats() -> Dict:
    return {
        'total_reads': 0,
        'filtered_unmapped': 0,
        'filtered_duplicate': 0,
        'filtered_secondary': 0,
        'filtered_supplementary': 0,
        'filtered_qcfail': 0,
        'filtered_low_mapq': 0,
        'skipped_read2': 0,
        'insert_size_fallback': 0,
        'reads_used': 0,
    }


def build_all_pileups(
    bam_files: List[str],
    chrom_sizes: Dict[str, int],
    bin_size: int,
    fragment_size: int,
    min_mapq: int = 20,
    min_fragment: int = _MIN_FRAGMENT,
    max_fragment: int = _MAX_FRAGMENT,
    blacklist_masks: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict]:
    """
    Build binned pileups for all chromosomes in a single sequential pass
    through each BAM file.

    Parameters
    ----------
    bam_files      : list of BAM/SAM paths (may be unsorted, no index required)
    chrom_sizes    : {chrom: size_bp} from BAM headers
    bin_size       : bin width in bp
    fragment_size  : SE extension length (also used as PE fallback)
    min_mapq       : minimum mapping quality; reads below this are discarded
    min_fragment   : minimum TLEN accepted for PE fragment interval
    max_fragment   : maximum TLEN accepted for PE fragment interval
    blacklist_masks: optional {chrom: bool_array} — blacklisted bins zeroed out

    Returns
    -------
    pileups     : {chrom: float32 ndarray, shape (n_bins,)}
    gc_per_bins : {chrom: float64 ndarray, shape (n_bins,)} — NaN for empty bins
    stats       : read-count dictionary (see _empty_stats)
    """
    # Pre-allocate buffers for every chromosome
    n_bins_map: Dict[str, int] = {
        c: (s + bin_size - 1) // bin_size for c, s in chrom_sizes.items()
    }
    pileups: Dict[str, np.ndarray] = {
        c: np.zeros(n, dtype=np.int32) for c, n in n_bins_map.items()
    }
    gc_sum: Dict[str, np.ndarray] = {
        c: np.zeros(n, dtype=np.float64) for c, n in n_bins_map.items()
    }
    gc_cnt: Dict[str, np.ndarray] = {
        c: np.zeros(n, dtype=np.int32) for c, n in n_bins_map.items()
    }

    stats = _empty_stats()

    for path in bam_files:
        mode = 'r' if path.endswith('.sam') else 'rb'
        try:
            with pysam.AlignmentFile(path, mode) as bam:
                for read in bam.fetch(until_eof=True):
                    stats['total_reads'] += 1

                    # --- QC filters ---
                    if read.is_unmapped:
                        stats['filtered_unmapped'] += 1
                        continue
                    if read.is_duplicate:
                        stats['filtered_duplicate'] += 1
                        continue
                    if read.is_secondary:
                        stats['filtered_secondary'] += 1
                        continue
                    if read.is_supplementary:
                        stats['filtered_supplementary'] += 1
                        continue
                    if read.is_qcfail:
                        stats['filtered_qcfail'] += 1
                        continue
                    if read.mapping_quality < min_mapq:
                        stats['filtered_low_mapq'] += 1
                        continue

                    # Skip R2 to avoid double-counting paired fragments
                    if read.is_paired and read.is_read2:
                        stats['skipped_read2'] += 1
                        continue

                    chrom = read.reference_name
                    if chrom not in chrom_sizes:
                        continue

                    chrom_size = chrom_sizes[chrom]
                    n_bins = n_bins_map[chrom]

                    start, end, used_fallback = _fragment_interval(
                        read, chrom_size, fragment_size, min_fragment, max_fragment
                    )
                    if used_fallback:
                        stats['insert_size_fallback'] += 1

                    s_bin = start // bin_size
                    e_bin = min(n_bins - 1, (end - 1) // bin_size)
                    if s_bin <= e_bin:
                        pileups[chrom][s_bin:e_bin + 1] += 1
                        stats['reads_used'] += 1

                    # Per-bin GC% from read sequence
                    seq = read.query_sequence
                    if seq:
                        seq_up = seq.upper()
                        gc = (seq_up.count('G') + seq_up.count('C')) / len(seq_up)
                        b = read.reference_start // bin_size
                        if 0 <= b < n_bins:
                            gc_sum[chrom][b] += gc
                            gc_cnt[chrom][b] += 1

        except Exception as exc:
            logger.warning(f'Error reading {path}: {exc}')

    # Apply blacklist masks and compute per-bin GC
    gc_per_bins: Dict[str, np.ndarray] = {}
    for chrom in chrom_sizes:
        if blacklist_masks and chrom in blacklist_masks:
            mask = blacklist_masks[chrom]
            pileups[chrom][mask] = 0
            gc_sum[chrom][mask] = 0
            gc_cnt[chrom][mask] = 0

        cnt = gc_cnt[chrom]
        gc_per_bins[chrom] = np.where(cnt > 0, gc_sum[chrom] / cnt, np.nan)

    out_pileups = {c: pileups[c].astype(np.float32) for c in chrom_sizes}
    return out_pileups, gc_per_bins, stats
