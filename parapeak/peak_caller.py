"""
Main peak-calling pipeline.

Pipeline stages
---------------
Stage 1 (parallel)  : per-chromosome pileup construction
Stage 2 (parallel)  : per-chromosome p-value computation
Stage 3 (sequential): genome-wide BH correction (synchronisation barrier)
Stage 4 (parallel)  : peak region calling per chromosome
"""
import logging
import math
import os
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

from parapeak.bam_reader import (
    build_pileup_for_chrom,
    count_mapped_reads,
    estimate_fragment_size,
    get_chromosome_sizes,
)
from parapeak.blacklist import build_blacklist_mask, parse_blacklist
from parapeak.gc_model import fit_gc_model, predict_gc
from parapeak.output import PeakRecord, write_narrowpeak, write_summit_bed, write_tsv
from parapeak.stats import benjamini_hochberg, nb_pvalues, zscore_pvalues

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker: pileup stage (runs in separate process)
# ---------------------------------------------------------------------------

def _pileup_worker(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Build treated and control pileups for one chromosome."""
    import numpy as np
    from parapeak.bam_reader import build_pileup_for_chrom
    from parapeak.blacklist import build_blacklist_mask, parse_blacklist

    chrom = kwargs['chrom']
    chrom_size = kwargs['chrom_size']
    treated_bams = kwargs['treated_bams']
    control_bams = kwargs['control_bams']
    blacklist_file = kwargs['blacklist_file']
    bin_size = kwargs['bin_size']
    fragment_size = kwargs['fragment_size']

    bl_regions = parse_blacklist(blacklist_file) if blacklist_file else None
    bl_mask = build_blacklist_mask(bl_regions, chrom, chrom_size, bin_size)

    treat_pileup, gc_per_bin, treat_reads = build_pileup_for_chrom(
        treated_bams, chrom, chrom_size, bin_size, fragment_size, bl_mask
    )

    if control_bams:
        ctrl_pileup, _, ctrl_reads = build_pileup_for_chrom(
            control_bams, chrom, chrom_size, bin_size, fragment_size, bl_mask
        )
    else:
        ctrl_pileup = treat_pileup.copy()
        ctrl_reads = treat_reads

    return {
        'chrom': chrom,
        'treat': treat_pileup,
        'ctrl': ctrl_pileup,
        'gc': gc_per_bin,
        'bl_mask': bl_mask,
        'treat_reads': treat_reads,
        'ctrl_reads': ctrl_reads,
    }


# ---------------------------------------------------------------------------
# Stage 2: per-chromosome p-values
# ---------------------------------------------------------------------------

def _compute_pvalues_for_chrom(
    chrom_data: Dict[str, Any],
    local_bins: int,
    scale_factor: float,
    gc_centers: np.ndarray,
    gc_mean_depth: np.ndarray,
    gc_var_depth: np.ndarray,
    pseudocount: float = 1.0,
) -> Dict[str, Any]:
    """Compute NB and Z-score p-values for one chromosome."""
    treat = chrom_data['treat'].astype(np.float64)
    ctrl = chrom_data['ctrl'].astype(np.float64)
    gc = chrom_data['gc']
    bl_mask = chrom_data['bl_mask']

    # --- NB p-values ---
    p_nb = nb_pvalues(
        observed=treat,
        ctrl_pileup=ctrl,
        local_bins=local_bins,
        scale_factor=scale_factor,
        blacklist_mask=bl_mask,
        pseudocount=pseudocount,
    )

    # --- GC-corrected Z-score p-values ---
    expected, variance = predict_gc(gc, gc_centers, gc_mean_depth, gc_var_depth)
    # Scale expected to match treatment depth
    expected_total = expected.sum()
    if expected_total > 0:
        expected = expected * (treat.sum() / expected_total)
        variance = variance * (treat.sum() / expected_total) ** 2

    p_z = zscore_pvalues(treat, expected, variance)

    # Blacklisted bins get p-value = 1 (never significant)
    p_nb[bl_mask] = 1.0
    p_z[bl_mask] = 1.0

    return {
        'chrom': chrom_data['chrom'],
        'p_nb': p_nb,
        'p_z': p_z,
        'treat': treat,
        'ctrl': ctrl,
        'bl_mask': bl_mask,
    }


# ---------------------------------------------------------------------------
# Stage 4: peak regions from significant-bin mask
# ---------------------------------------------------------------------------

def _call_peaks_for_chrom(
    chrom: str,
    treat: np.ndarray,
    ctrl: np.ndarray,
    q_nb: np.ndarray,
    q_z: np.ndarray,
    p_nb: np.ndarray,
    p_z: np.ndarray,
    bl_mask: np.ndarray,
    bin_size: int,
    qvalue_thr: float,
    min_length_bins: int,
    max_gap_bins: int,
    scale_factor: float,
    pseudocount: float,
    min_count: float,
    min_fold: float,
    peak_prefix: str,
    chrom_offset: int,
) -> List[PeakRecord]:
    """
    Identify contiguous runs of significant bins, merge across small gaps,
    then apply signal-strength filters before building PeakRecord objects.

    Signal filters (applied after statistical significance)
    -------------------------------------------------------
    min_count : minimum pileup value at the summit bin.
        At low coverage (MiSeq), statistical tests can pass bins with only
        1-2 reads if the control happens to have zero reads there.  This
        absolute floor removes those artefacts regardless of p-value.

    min_fold  : minimum fold enrichment over the pseudocount-corrected
        background.  fold = (mean_treat + pc) / (mean_ctrl_scaled + pc)
        where pc = pseudocount / local_bins.  Using pseudocount in the
        denominator prevents infinite fold values when control = 0 and
        gives a meaningful enrichment estimate at all coverage depths.
    """
    sig = (q_nb < qvalue_thr) & (q_z < qvalue_thr) & (~bl_mask)
    n_bins = len(sig)

    if not sig.any():
        return []

    # Expand gap: allow up to max_gap_bins non-significant bins between sig regions
    if max_gap_bins > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones(2 * max_gap_bins + 1, dtype=bool)
        sig = binary_dilation(sig, structure=struct) & ~bl_mask

    # Pseudocount per bin for fold calculation (same units as pileup)
    # We store pseudocount as reads-per-local-window; distribute evenly.
    # Use a rough local_bins estimate from local_window implied by the
    # caller (not available here), so fall back to pseudocount directly
    # as a per-region floor on the scaled control mean.
    pc = pseudocount  # per-region pseudocount (added to both numerator & denominator)

    peaks: List[PeakRecord] = []
    idx = 0
    peak_count = 0

    while idx < n_bins:
        if not sig[idx]:
            idx += 1
            continue
        # Start of a significant run
        start_bin = idx
        while idx < n_bins and sig[idx]:
            idx += 1
        end_bin = idx  # exclusive

        length_bins = end_bin - start_bin
        if length_bins < min_length_bins:
            continue

        # Summit: bin with maximum treatment signal in the region
        region_treat = treat[start_bin:end_bin]
        summit_val = float(np.max(region_treat))

        # --- Signal filter 1: minimum absolute count at summit ---
        if summit_val < min_count:
            continue

        mean_treat = float(np.mean(region_treat))
        mean_ctrl_scaled = float(np.mean(ctrl[start_bin:end_bin])) * scale_factor

        # Fold enrichment with pseudocount (no infinite values)
        fold_enrich = (mean_treat + pc) / (mean_ctrl_scaled + pc)

        # --- Signal filter 2: minimum fold enrichment ---
        if min_fold > 1.0 and fold_enrich < min_fold:
            continue

        start_bp = start_bin * bin_size
        end_bp = end_bin * bin_size
        summit_rel = int(np.argmax(region_treat))
        summit_abs = (start_bin + summit_rel) * bin_size + bin_size // 2

        region_q_nb = q_nb[start_bin:end_bin]
        region_q_z = q_z[start_bin:end_bin]
        region_p_nb = p_nb[start_bin:end_bin]
        region_p_z = p_z[start_bin:end_bin]

        min_q_nb = float(np.min(region_q_nb))
        min_q_z  = float(np.min(region_q_z))
        min_p_nb = float(np.min(region_p_nb))
        min_p_z  = float(np.min(region_p_z))

        peak_count += 1
        name = f'{peak_prefix}_{chrom_offset + peak_count}'
        peaks.append(PeakRecord(
            chrom=chrom,
            start=start_bp,
            end=end_bp,
            name=name,
            fold_enrichment=fold_enrich,
            neg_log10_pval_nb=-math.log10(max(min_p_nb, 1e-300)),
            neg_log10_pval_z=-math.log10(max(min_p_z, 1e-300)),
            neg_log10_qval_nb=-math.log10(max(min_q_nb, 1e-300)),
            neg_log10_qval_z=-math.log10(max(min_q_z, 1e-300)),
            summit_offset=summit_abs - start_bp,
        ))

    return peaks


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_peak_calling(args: Namespace) -> None:
    """Orchestrate the full peak-calling pipeline."""
    os.makedirs(args.output, exist_ok=True)
    logger.info(f'Output directory: {args.output}')

    # --- genome sizes ---
    all_bams = args.treated + (args.control or [])
    chrom_sizes = get_chromosome_sizes(all_bams)
    # Filter out chromosomes with no data
    valid_chroms = {c: s for c, s in chrom_sizes.items() if s > 0}
    logger.info(f'Found {len(valid_chroms)} chromosomes in BAM headers')

    # --- fragment size ---
    if args.fragment_size:
        fragment_size = args.fragment_size
    else:
        fragment_size = estimate_fragment_size(args.treated)
    logger.info(f'Fragment size: {fragment_size} bp')

    # --- library sizes ---
    treat_total = count_mapped_reads(args.treated)
    ctrl_total = count_mapped_reads(args.control) if args.control else treat_total
    if ctrl_total > 0 and treat_total > 0:
        scale_factor = treat_total / ctrl_total
    else:
        scale_factor = 1.0
    logger.info(
        f'Reads: treated={treat_total:,}, control={ctrl_total:,}, '
        f'scale_factor={scale_factor:.4f}'
    )

    local_bins = max(1, args.local_window // args.bin_size)
    min_length_bins = max(1, args.min_length // args.bin_size)
    max_gap_bins = max(0, args.max_gap // args.bin_size)
    logger.info(
        f'Signal filters: min-count={args.min_count}, '
        f'min-fold={args.min_fold}, pseudocount={args.pseudocount}'
    )

    # -----------------------------------------------------------------------
    # Stage 1: pileup (parallel)
    # -----------------------------------------------------------------------
    chrom_items = sorted(valid_chroms.items(), key=lambda x: x[1], reverse=True)
    worker_kwargs = [
        {
            'chrom': chrom,
            'chrom_size': size,
            'treated_bams': args.treated,
            'control_bams': args.control,
            'blacklist_file': args.blacklist,
            'bin_size': args.bin_size,
            'fragment_size': fragment_size,
        }
        for chrom, size in chrom_items
    ]

    logger.info(f'Building pileups for {len(chrom_items)} chromosomes '
                f'using {args.threads} worker(s)')
    chrom_results: Dict[str, Dict] = {}

    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(_pileup_worker, kw): kw['chrom'] for kw in worker_kwargs}
        for fut in as_completed(futures):
            chrom = futures[fut]
            try:
                result = fut.result()
                chrom_results[chrom] = result
                logger.info(
                    f'  {chrom}: {result["treat_reads"]:,} treated reads, '
                    f'{result["ctrl_reads"]:,} control reads'
                )
            except Exception as exc:
                logger.warning(f'  Skipping {chrom}: {exc}')

    if not chrom_results:
        logger.error('No chromosomes processed successfully; exiting')
        return

    # -----------------------------------------------------------------------
    # Stage 2-pre: GC model (requires all chromosomes)
    # -----------------------------------------------------------------------
    pileup_list = [chrom_results[c]['treat'] for c in chrom_results]
    gc_list = [chrom_results[c]['gc'] for c in chrom_results]
    gc_centers, gc_mean_depth, gc_var_depth = fit_gc_model(pileup_list, gc_list)

    # -----------------------------------------------------------------------
    # Stage 2: p-values (parallel)
    # -----------------------------------------------------------------------
    logger.info('Computing p-values')
    chrom_pvals: Dict[str, Dict] = {}

    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        futures = {}
        for chrom, data in chrom_results.items():
            fut = pool.submit(
                _compute_pvalues_for_chrom,
                data, local_bins, scale_factor,
                gc_centers, gc_mean_depth, gc_var_depth,
                args.pseudocount,
            )
            futures[fut] = chrom

        for fut in as_completed(futures):
            chrom = futures[fut]
            try:
                chrom_pvals[chrom] = fut.result()
            except Exception as exc:
                logger.warning(f'  P-value computation failed for {chrom}: {exc}')

    # -----------------------------------------------------------------------
    # Stage 3: genome-wide BH correction (sequential)
    # -----------------------------------------------------------------------
    logger.info('Applying Benjamini-Hochberg correction genome-wide')
    chrom_order = sorted(chrom_pvals.keys())

    # Concatenate all p-values in chromosome order
    p_nb_all = np.concatenate([chrom_pvals[c]['p_nb'] for c in chrom_order])
    p_z_all = np.concatenate([chrom_pvals[c]['p_z'] for c in chrom_order])

    q_nb_all = benjamini_hochberg(p_nb_all)
    q_z_all = benjamini_hochberg(p_z_all)

    # Split back per chromosome
    split_points: List[int] = []
    pos = 0
    for c in chrom_order:
        n = len(chrom_pvals[c]['p_nb'])
        split_points.append(pos)
        pos += n
    split_points.append(pos)

    for i, c in enumerate(chrom_order):
        lo, hi = split_points[i], split_points[i + 1]
        chrom_pvals[c]['q_nb'] = q_nb_all[lo:hi]
        chrom_pvals[c]['q_z'] = q_z_all[lo:hi]

    # -----------------------------------------------------------------------
    # Stage 4: peak calling (parallel)
    # -----------------------------------------------------------------------
    logger.info('Calling peaks')
    all_peaks: List[PeakRecord] = []

    # Assign offsets so peak names are globally unique
    chrom_peak_offset: Dict[str, int] = {}
    cumulative = 0
    for c in chrom_order:
        chrom_peak_offset[c] = cumulative
        cumulative += 10_000  # generous upper bound per chrom

    with ProcessPoolExecutor(max_workers=args.threads) as pool:
        futures = {}
        for c in chrom_order:
            pv = chrom_pvals[c]
            fut = pool.submit(
                _call_peaks_for_chrom,
                chrom=c,
                treat=pv['treat'],
                ctrl=pv['ctrl'],
                q_nb=pv['q_nb'],
                q_z=pv['q_z'],
                p_nb=pv['p_nb'],
                p_z=pv['p_z'],
                bl_mask=pv['bl_mask'],
                bin_size=args.bin_size,
                qvalue_thr=args.qvalue,
                min_length_bins=min_length_bins,
                max_gap_bins=max_gap_bins,
                scale_factor=scale_factor,
                pseudocount=args.pseudocount,
                min_count=args.min_count,
                min_fold=args.min_fold,
                peak_prefix=args.name,
                chrom_offset=chrom_peak_offset[c],
            )
            futures[fut] = c

        for fut in as_completed(futures):
            c = futures[fut]
            try:
                peaks = fut.result()
                all_peaks.extend(peaks)
                logger.info(f'  {c}: {len(peaks)} peaks')
            except Exception as exc:
                logger.warning(f'  Peak calling failed for {c}: {exc}')

    # Sort by chromosome then start
    all_peaks.sort(key=lambda pk: (pk.chrom, pk.start))

    logger.info(f'Total peaks called: {len(all_peaks)}')

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    np_path = os.path.join(args.output, f'{args.name}_peaks.narrowPeak')
    sb_path = os.path.join(args.output, f'{args.name}_summits.bed')
    ts_path = os.path.join(args.output, f'{args.name}_peaks.tsv')

    write_narrowpeak(all_peaks, np_path)
    write_summit_bed(all_peaks, sb_path)
    write_tsv(all_peaks, ts_path)

    logger.info(f'Written:\n  {np_path}\n  {sb_path}\n  {ts_path}')
