"""
BAM/SAM reading utilities: pileup construction and fragment-size estimation.

The pileup is stored as a numpy int32 array (one value per bin).
MACS3's linked-list approach is intentionally NOT used; numpy array
accumulators are simpler, parallelisable per chromosome, and fast enough
because the true bottleneck is BAM file I/O and gzip decompression.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pysam

logger = logging.getLogger(__name__)


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
    Estimate fragment size from paired-end insert sizes.
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
                if read.is_proper_pair and read.template_length > 0:
                    inserts.append(abs(read.template_length))

    if not inserts:
        logger.warning('No paired-end reads found; using default fragment size of 200 bp')
        return 200

    median = int(np.median(inserts))
    logger.info(f'Estimated fragment size: {median} bp (from {len(inserts)} paired reads)')
    return median


def count_mapped_reads(bam_files: List[str]) -> int:
    """Count total mapped reads (uses index statistics when available)."""
    total = 0
    for path in bam_files:
        mode = 'r' if path.endswith('.sam') else 'rb'
        with pysam.AlignmentFile(path, mode) as bam:
            try:
                stats = bam.get_index_statistics()
                total += sum(s.mapped for s in stats)
            except ValueError:
                total += sum(1 for r in bam.fetch() if not r.is_unmapped)
    return total


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

    Reads are extended from their 5' end by *fragment_size* bp.
    The bottleneck is BAM decompression, not Python arithmetic, so a
    plain numpy int32 accumulator is the right data structure here.

    Returns
    -------
    pileup    : np.ndarray[int32], shape (n_bins,)
    gc_per_bin: np.ndarray[float64], shape (n_bins,)  – NaN for empty bins
    n_reads   : int
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

                    if read.is_reverse:
                        end = read.reference_end or (
                            read.reference_start + (read.query_length or 1)
                        )
                        start = max(0, end - fragment_size)
                    else:
                        start = read.reference_start
                        end = min(chrom_size, start + fragment_size)

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
