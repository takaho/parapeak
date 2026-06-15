"""
Output writers for parapeak results.

Formats
-------
narrowPeak (ENCODE BED6+4)
    chrom  start  end  name  score  strand  signalValue  pValue  qValue  peak

summit BED
    chrom  start  end  name  score

JSON run report
    Settings, QC statistics, and summary counts.
"""
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np


@dataclass
class PeakRecord:
    chrom: str
    start: int           # 0-based
    end: int             # exclusive
    name: str
    fold_enrichment: float
    neg_log10_pval_nb: float
    neg_log10_pval_z: float
    neg_log10_qval_nb: float
    neg_log10_qval_z: float
    summit_offset: int   # relative to start

    @property
    def score(self) -> int:
        """UCSC score field: min(-10*log10(min_qvalue), 1000)."""
        min_q = min(self.neg_log10_qval_nb, self.neg_log10_qval_z)
        return min(int(min_q * 10), 1000)

    @property
    def length(self) -> int:
        return self.end - self.start


def write_narrowpeak(peaks: List[PeakRecord], path: str) -> None:
    """Write ENCODE narrowPeak file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        for pk in peaks:
            min_qval = min(pk.neg_log10_qval_nb, pk.neg_log10_qval_z)
            min_pval = min(pk.neg_log10_pval_nb, pk.neg_log10_pval_z)
            fh.write(
                f'{pk.chrom}\t{pk.start}\t{pk.end}\t{pk.name}\t'
                f'{pk.score}\t.\t{pk.fold_enrichment:.4f}\t'
                f'{min_pval:.4f}\t{min_qval:.4f}\t{pk.summit_offset}\n'
            )


def write_summit_bed(peaks: List[PeakRecord], path: str) -> None:
    """Write summit positions as a BED file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        for pk in peaks:
            summit = pk.start + pk.summit_offset
            fh.write(
                f'{pk.chrom}\t{summit}\t{summit + 1}\t{pk.name}\t{pk.score}\n'
            )


def write_tsv(peaks: List[PeakRecord], path: str) -> None:
    """Write detailed TSV with all score columns."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    header = (
        'chrom\tstart\tend\tname\tlength\tsummit\t'
        'fold_enrichment\t'
        '-log10(p_nb)\t-log10(p_z)\t'
        '-log10(q_nb)\t-log10(q_z)\n'
    )
    with open(path, 'w') as fh:
        fh.write(header)
        for pk in peaks:
            fh.write(
                f'{pk.chrom}\t{pk.start}\t{pk.end}\t{pk.name}\t'
                f'{pk.length}\t{pk.start + pk.summit_offset}\t'
                f'{pk.fold_enrichment:.4f}\t'
                f'{pk.neg_log10_pval_nb:.4f}\t{pk.neg_log10_pval_z:.4f}\t'
                f'{pk.neg_log10_qval_nb:.4f}\t{pk.neg_log10_qval_z:.4f}\n'
            )


def write_json(
    path: str,
    args: Any,
    fragment_size_used: int,
    scale_factor: float,
    treat_stats: Dict,
    ctrl_stats: Optional[Dict],
    peaks_called: int,
) -> None:
    """
    Write a human-readable JSON run report containing settings and statistics.

    Parameters
    ----------
    path              : output file path
    args              : parsed argparse Namespace
    fragment_size_used: actual fragment size used (estimated or user-supplied)
    scale_factor      : treat/ctrl normalisation factor
    treat_stats       : read-count dict from build_all_pileups for treated files
    ctrl_stats        : read-count dict from build_all_pileups for control files,
                        or None when no control was provided
    peaks_called      : total number of peaks in the final output
    """
    settings: Dict[str, Any] = {
        'treated': args.treated,
        'control': args.control,
        'blacklist': args.blacklist,
        'output': args.output,
        'name': args.name,
        'bin_size': args.bin_size,
        'fragment_size': args.fragment_size,
        'min_mapq': args.min_mapq,
        'min_fragment': args.min_fragment,
        'max_fragment': args.max_fragment,
        'local_window': args.local_window,
        'qvalue': args.qvalue,
        'min_length': args.min_length,
        'max_gap': args.max_gap,
        'min_count': args.min_count,
        'min_fold': args.min_fold,
        'pseudocount': args.pseudocount,
        'threads': args.threads,
        'bedgraph_value': args.bedgraph_value,
    }

    report = {
        'run': {
            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'tool': 'parapeak',
        },
        'settings': settings,
        'statistics': {
            'fragment_size_used': fragment_size_used,
            'scale_factor': round(scale_factor, 6),
            'treated': treat_stats,
            'control': ctrl_stats,
            'peaks_called': peaks_called,
        },
    }

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(report, fh, indent=2)
        fh.write('\n')


# ---------------------------------------------------------------------------
# bedGraph output
# ---------------------------------------------------------------------------

_BEDGRAPH_LABELS = {
    'fold_enrichment': 'Fold enrichment (treated / scaled control)',
    'pvalue':          '-log10(p-value)',
    'pileup':          'Treated pileup (read count per bin)',
}


def _iter_bedgraph_records(
    values: np.ndarray, bin_size: int, chrom_size: int
) -> Generator[Tuple[int, int, float], None, None]:
    """Yield (start, end, value) bedGraph records, merging consecutive equal-valued bins.

    Bins with value NaN or 0 are skipped (no record emitted).
    """
    rounded = np.round(values.astype(np.float64), 4)
    n = len(rounded)
    if n == 0:
        return

    valid = ~np.isnan(rounded) & (rounded != 0.0)

    breaks = np.zeros(n, dtype=bool)
    breaks[0] = True
    # New segment when valid-status changes OR both valid but value differs
    breaks[1:] = (valid[1:] != valid[:-1]) | (
        valid[1:] & valid[:-1] & (rounded[1:] != rounded[:-1])
    )

    seg_starts = np.where(breaks)[0]
    seg_ends = np.append(seg_starts[1:], n)

    for s, e in zip(seg_starts, seg_ends):
        if not valid[s]:
            continue
        yield (
            int(s) * bin_size,
            min(int(e) * bin_size, chrom_size),
            float(rounded[s]),
        )


def write_bedgraph(
    path: str,
    chrom_values: Dict[str, np.ndarray],
    chrom_sizes: Dict[str, int],
    bin_size: int,
    chrom_order: List[str],
    name: str = 'parapeak',
    value_type: str = 'fold_enrichment',
) -> None:
    """Write a genome-wide bedGraph file.

    Parameters
    ----------
    chrom_values : per-chromosome float64 arrays (NaN = blacklisted / skip)
    chrom_sizes  : chromosome lengths in bp (used to clip the last bin end)
    bin_size     : bin width in bp
    chrom_order  : output chromosome ordering
    value_type   : one of 'fold_enrichment', 'pvalue', 'pileup'
    """
    label = _BEDGRAPH_LABELS.get(value_type, value_type)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        fh.write(f'track type=bedGraph name="{name}" description="{label}"\n')
        for chrom in chrom_order:
            if chrom not in chrom_values or chrom not in chrom_sizes:
                continue
            for start, end, val in _iter_bedgraph_records(
                chrom_values[chrom], bin_size, chrom_sizes[chrom]
            ):
                fh.write(f'{chrom}\t{start}\t{end}\t{val:.4f}\n')
