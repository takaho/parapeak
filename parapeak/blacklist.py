"""
Parse blacklist BED files and provide fast interval overlap queries.
"""
import gzip
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def parse_blacklist(bed_file: str) -> Dict[str, List[Tuple[int, int]]]:
    """Return {chrom: [(start, end), ...]} from a BED or BED.gz file."""
    regions: Dict[str, List[Tuple[int, int]]] = {}
    opener = gzip.open if bed_file.endswith('.gz') else open

    with opener(bed_file, 'rt') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            chrom = parts[0]
            try:
                start, end = int(parts[1]), int(parts[2])
            except ValueError:
                continue
            regions.setdefault(chrom, []).append((start, end))

    n_regions = sum(len(v) for v in regions.values())
    logger.info(f'Loaded {n_regions} blacklist regions from {bed_file}')
    return regions


def build_blacklist_mask(
    regions: Optional[Dict[str, List[Tuple[int, int]]]],
    chrom: str,
    chrom_size: int,
    bin_size: int,
) -> np.ndarray:
    """
    Return a boolean mask of length n_bins where True means the bin
    overlaps at least one blacklist region.
    """
    n_bins = (chrom_size + bin_size - 1) // bin_size
    mask = np.zeros(n_bins, dtype=bool)

    if regions is None or chrom not in regions:
        return mask

    for start, end in regions[chrom]:
        s_bin = max(0, start // bin_size)
        e_bin = min(n_bins - 1, (end - 1) // bin_size)
        if s_bin <= e_bin:
            mask[s_bin : e_bin + 1] = True

    return mask
