"""
Main peak-calling pipeline.

Pipeline stages
---------------
Stage 1 (sequential) : single-pass pileup construction for all chromosomes
Stage 2 (parallel)   : per-chromosome p-value computation
Stage 3 (sequential) : genome-wide BH correction (synchronisation barrier)
Stage 4 (parallel)   : peak region calling per chromosome

Stage 1 reads each BAM file exactly once regardless of sort order, making
the pipeline suitable for unsorted BAM files.  Downstream stages 2 and 4
remain fully parallelised across chromosomes.
"""
import logging
import math
import os
from argparse import Namespace
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

from parapeak.bam_reader import (
    build_all_pileups,
    estimate_fragment_size,
    get_chromosome_sizes,
)
from parapeak.blacklist import build_blacklist_mask, parse_blacklist
from parapeak.gc_model import fit_gc_model, predict_gc
from parapeak.output import (
    PeakRecord,
    write_bedgraph,
    write_json,
    write_narrowpeak,
    write_summit_bed,
    write_tsv,
)
from parapeak.stats import benjamini_hochberg, nb_pvalues, zscore_pvalues

logger = logging.getLogger(__name__)


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

    p_nb = nb_pvalues(
        observed=treat,
        ctrl_pileup=ctrl,
        local_bins=local_bins,
        scale_factor=scale_factor,
        blacklist_mask=bl_mask,
        pseudocount=pseudocount,
    )

    expected, variance = predict_gc(gc, gc_centers, gc_mean_depth, gc_var_depth)
    expected_total = expected.sum()
    if expected_total > 0:
        expected = expected * (treat.sum() / expected_total)
        variance = variance * (treat.sum() / expected_total) ** 2

    p_z = zscore_pvalues(treat, expected, variance)

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
    """
    sig = (q_nb < qvalue_thr) & (q_z < qvalue_thr) & (~bl_mask)
    n_bins = len(sig)

    if not sig.any():
        return []

    if max_gap_bins > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones(2 * max_gap_bins + 1, dtype=bool)
        sig = binary_dilation(sig, structure=struct) & ~bl_mask

    pc = pseudocount

    peaks: List[PeakRecord] = []
    idx = 0
    peak_count = 0

    while idx < n_bins:
        if not sig[idx]:
            idx += 1
            continue
        start_bin = idx
        while idx < n_bins and sig[idx]:
            idx += 1
        end_bin = idx

        if end_bin - start_bin < min_length_bins:
            continue

        region_treat = treat[start_bin:end_bin]
        summit_val = float(np.max(region_treat))

        if summit_val < min_count:
            continue

        mean_treat = float(np.mean(region_treat))
        mean_ctrl_scaled = float(np.mean(ctrl[start_bin:end_bin])) * scale_factor
        fold_enrich = (mean_treat + pc) / (mean_ctrl_scaled + pc)

        if min_fold > 1.0 and fold_enrich < min_fold:
            continue

        start_bp = start_bin * bin_size
        end_bp = end_bin * bin_size
        summit_rel = int(np.argmax(region_treat))
        summit_abs = (start_bin + summit_rel) * bin_size + bin_size // 2

        region_q_nb = q_nb[start_bin:end_bin]
        region_q_z  = q_z[start_bin:end_bin]
        region_p_nb = p_nb[start_bin:end_bin]
        region_p_z  = p_z[start_bin:end_bin]

        min_q_nb = float(np.min(region_q_nb))
        min_q_z  = float(np.min(region_q_z))
        min_p_nb = float(np.min(region_p_nb))
        min_p_z  = float(np.min(region_p_z))

        peak_count += 1
        peaks.append(PeakRecord(
            chrom=chrom,
            start=start_bp,
            end=end_bp,
            name=f'{peak_prefix}_{chrom_offset + peak_count}',
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
    valid_chroms = {c: s for c, s in chrom_sizes.items() if s > 0}
    logger.info(f'Found {len(valid_chroms)} chromosomes in BAM headers')

    # --- fragment size ---
    if args.fragment_size:
        fragment_size = args.fragment_size
    else:
        fragment_size = estimate_fragment_size(
            args.treated,
            min_fragment=args.min_fragment,
            max_fragment=args.max_fragment,
        )
    logger.info(f'Fragment size: {fragment_size} bp')

    # --- blacklist masks (built once, shared across pileup calls) ---
    bl_regions = parse_blacklist(args.blacklist) if args.blacklist else None
    blacklist_masks = {
        chrom: build_blacklist_mask(bl_regions, chrom, size, args.bin_size)
        for chrom, size in valid_chroms.items()
    }

    local_bins = max(1, args.local_window // args.bin_size)
    min_length_bins = max(1, args.min_length // args.bin_size)
    max_gap_bins = max(0, args.max_gap // args.bin_size)

    # -----------------------------------------------------------------------
    # Stage 1: single-pass pileup construction
    # -----------------------------------------------------------------------
    logger.info(f'Building treated pileups (single pass over {len(args.treated)} file(s))')
    treat_pileups, treat_gc, treat_stats = build_all_pileups(
        bam_files=args.treated,
        chrom_sizes=valid_chroms,
        bin_size=args.bin_size,
        fragment_size=fragment_size,
        min_mapq=args.min_mapq,
        min_fragment=args.min_fragment,
        max_fragment=args.max_fragment,
        blacklist_masks=blacklist_masks,
    )
    logger.info(
        f'  treated: {treat_stats["total_reads"]:,} total reads, '
        f'{treat_stats["reads_used"]:,} used'
    )

    if args.control:
        logger.info(
            f'Building control pileups (single pass over {len(args.control)} file(s))'
        )
        ctrl_pileups, ctrl_gc, ctrl_stats = build_all_pileups(
            bam_files=args.control,
            chrom_sizes=valid_chroms,
            bin_size=args.bin_size,
            fragment_size=fragment_size,
            min_mapq=args.min_mapq,
            min_fragment=args.min_fragment,
            max_fragment=args.max_fragment,
            blacklist_masks=blacklist_masks,
        )
        logger.info(
            f'  control: {ctrl_stats["total_reads"]:,} total reads, '
            f'{ctrl_stats["reads_used"]:,} used'
        )
    else:
        ctrl_pileups = {c: treat_pileups[c].copy() for c in treat_pileups}
        ctrl_gc = treat_gc
        ctrl_stats = None

    treat_total = treat_stats['reads_used']
    ctrl_total = ctrl_stats['reads_used'] if ctrl_stats else treat_total
    scale_factor = treat_total / ctrl_total if ctrl_total > 0 and treat_total > 0 else 1.0
    logger.info(
        f'Reads used: treated={treat_total:,}, control={ctrl_total:,}, '
        f'scale_factor={scale_factor:.4f}'
    )

    # Assemble per-chromosome data dict
    chrom_results: Dict[str, Dict] = {}
    for chrom in valid_chroms:
        if chrom not in treat_pileups:
            continue
        chrom_results[chrom] = {
            'chrom': chrom,
            'treat': treat_pileups[chrom],
            'ctrl': ctrl_pileups[chrom],
            'gc': treat_gc[chrom],
            'bl_mask': blacklist_masks[chrom],
        }

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

    p_nb_all = np.concatenate([chrom_pvals[c]['p_nb'] for c in chrom_order])
    p_z_all  = np.concatenate([chrom_pvals[c]['p_z']  for c in chrom_order])

    q_nb_all = benjamini_hochberg(p_nb_all)
    q_z_all  = benjamini_hochberg(p_z_all)

    pos = 0
    split_points: List[int] = []
    for c in chrom_order:
        split_points.append(pos)
        pos += len(chrom_pvals[c]['p_nb'])
    split_points.append(pos)

    for i, c in enumerate(chrom_order):
        lo, hi = split_points[i], split_points[i + 1]
        chrom_pvals[c]['q_nb'] = q_nb_all[lo:hi]
        chrom_pvals[c]['q_z']  = q_z_all[lo:hi]

    # -----------------------------------------------------------------------
    # Stage 4: peak calling (parallel)
    # -----------------------------------------------------------------------
    logger.info('Calling peaks')
    all_peaks: List[PeakRecord] = []

    chrom_peak_offset: Dict[str, int] = {}
    cumulative = 0
    for c in chrom_order:
        chrom_peak_offset[c] = cumulative
        cumulative += 10_000

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

    all_peaks.sort(key=lambda pk: (pk.chrom, pk.start))
    logger.info(f'Total peaks called: {len(all_peaks)}')

    # -----------------------------------------------------------------------
    # Write output
    # -----------------------------------------------------------------------
    np_path   = os.path.join(args.output, f'{args.name}_peaks.narrowPeak')
    sb_path   = os.path.join(args.output, f'{args.name}_summits.bed')
    ts_path   = os.path.join(args.output, f'{args.name}_peaks.tsv')
    json_path = os.path.join(args.output, f'{args.name}_run.json')
    bg_path   = os.path.join(args.output, f'{args.name}.bedGraph')

    write_narrowpeak(all_peaks, np_path)
    write_summit_bed(all_peaks, sb_path)
    write_tsv(all_peaks, ts_path)
    write_json(
        path=json_path,
        args=args,
        fragment_size_used=fragment_size,
        scale_factor=scale_factor,
        treat_stats=treat_stats,
        ctrl_stats=ctrl_stats,
        peaks_called=len(all_peaks),
    )

    # Build per-chromosome value arrays for bedGraph
    bg_values: Dict[str, np.ndarray] = {}
    vtype = args.bedgraph_value
    for c in chrom_order:
        if c not in chrom_pvals:
            continue
        pv = chrom_pvals[c]
        treat_arr = pv['treat']
        ctrl_arr  = pv['ctrl']
        bl_mask   = pv['bl_mask']

        if vtype == 'pileup':
            vals = treat_arr.astype(np.float64)
        elif vtype == 'pvalue':
            min_p = np.minimum(pv['p_nb'], pv['p_z'])
            vals = -np.log10(np.maximum(min_p, 1e-300))
        else:  # fold_enrichment
            pc = args.pseudocount
            vals = (treat_arr + pc) / (ctrl_arr * scale_factor + pc)

        vals = vals.astype(np.float64)
        vals[bl_mask] = np.nan
        bg_values[c] = vals

    write_bedgraph(
        path=bg_path,
        chrom_values=bg_values,
        chrom_sizes=valid_chroms,
        bin_size=args.bin_size,
        chrom_order=chrom_order,
        name=args.name,
        value_type=vtype,
    )

    logger.info(
        f'Written:\n  {np_path}\n  {sb_path}\n  {ts_path}\n  {json_path}\n  {bg_path}'
    )
