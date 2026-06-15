"""
Output writers for parapeak results.

Formats
-------
narrowPeak (ENCODE BED6+4)
    chrom  start  end  name  score  strand  signalValue  pValue  qValue  peak

summit BED
    chrom  start  end  name  score

signal BED (BED4)
    chrom  start  end  value   (significant peaks only)

bigWig pileup
    per-bin pileup minus genome mean, positive values only

JSON run report
    Settings, QC statistics, and summary counts.
"""
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

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
    summit_pileup: float

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
    """Write a human-readable JSON run report containing settings and statistics."""
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
        'signal_value': args.signal_value,
        'bigwig_bin': args.bigwig_bin,
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
# Signal BED output (significant peaks only)
# ---------------------------------------------------------------------------

def write_peak_signal_bed(
    peaks: List[PeakRecord],
    path: str,
    value_type: str = 'fold_enrichment',
) -> None:
    """Write a BED4 file with one row per significant peak.

    Parameters
    ----------
    peaks      : list of called peaks
    path       : output file path
    value_type : 'fold_enrichment', 'pvalue', or 'pileup'
    """
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as fh:
        for pk in peaks:
            if value_type == 'pileup':
                val = pk.summit_pileup
            elif value_type == 'pvalue':
                val = max(pk.neg_log10_pval_nb, pk.neg_log10_pval_z)
            else:  # fold_enrichment
                val = pk.fold_enrichment
            fh.write(f'{pk.chrom}\t{pk.start}\t{pk.end}\t{val:.4f}\n')


# ---------------------------------------------------------------------------
# bigWig pileup output
# ---------------------------------------------------------------------------

def write_pileup_bigwig(
    path: str,
    chrom_pileups: Dict[str, np.ndarray],
    chrom_sizes: Dict[str, int],
    chrom_order: List[str],
    bin_size: int,
    bigwig_bin: int,
    bl_masks: Dict[str, np.ndarray],
) -> None:
    """Write a bigWig of pileup minus genome-wide mean, positive values only.

    Parameters
    ----------
    chrom_pileups : per-chromosome treated read-count arrays at bin_size resolution
    chrom_sizes   : chromosome lengths in bp
    chrom_order   : output chromosome ordering (must match bigWig header)
    bin_size      : resolution of chrom_pileups in bp
    bigwig_bin    : output bin size in bp (default 25)
    bl_masks      : per-chromosome boolean blacklist masks
    """
    try:
        import pyBigWig
    except ImportError:
        raise ImportError(
            'pyBigWig is required for bigWig output. '
            'Install it with: pip install pyBigWig'
        )

    # Genome-wide mean from all non-blacklisted bins
    total_sum = 0.0
    total_count = 0
    for c in chrom_order:
        if c not in chrom_pileups:
            continue
        arr = chrom_pileups[c].astype(np.float64)
        mask = bl_masks.get(c, np.zeros(len(arr), dtype=bool))
        valid = arr[~mask]
        total_sum += float(valid.sum())
        total_count += len(valid)
    global_mean = total_sum / total_count if total_count > 0 else 0.0

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    bw = pyBigWig.open(path, 'w')
    header = [(c, chrom_sizes[c]) for c in chrom_order if c in chrom_sizes]
    bw.addHeader(header)

    for c in chrom_order:
        if c not in chrom_pileups or c not in chrom_sizes:
            continue
        chrom_len = chrom_sizes[c]
        arr = chrom_pileups[c].astype(np.float64)
        mask = bl_masks.get(c, np.zeros(len(arr), dtype=bool))
        arr[mask] = np.nan

        n_new = math.ceil(chrom_len / bigwig_bin)
        new_sum = np.zeros(n_new, dtype=np.float64)
        new_cnt = np.zeros(n_new, dtype=np.int64)

        orig_starts = np.arange(len(arr), dtype=np.int64) * bin_size
        new_idx = np.minimum(orig_starts // bigwig_bin, n_new - 1)

        valid_mask = ~np.isnan(arr)
        np.add.at(new_sum, new_idx[valid_mask], arr[valid_mask])
        np.add.at(new_cnt, new_idx[valid_mask], 1)

        with np.errstate(invalid='ignore'):
            new_mean = np.where(new_cnt > 0, new_sum / new_cnt, 0.0)

        signal = new_mean - global_mean
        pos_mask = signal > 0.0
        if not pos_mask.any():
            continue

        j_idx = np.where(pos_mask)[0]
        starts = (j_idx * bigwig_bin).tolist()
        ends = [min(int(j + 1) * bigwig_bin, chrom_len) for j in j_idx]
        vals = [float(v) for v in signal[pos_mask]]

        bw.addEntries(c, starts, ends=ends, values=vals)

    bw.close()
