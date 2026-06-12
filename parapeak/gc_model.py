"""
GC-content correction model.

Builds a linear interpolation model:  GC% → expected read depth
from per-bin (GC%, observed_depth) pairs pooled across all chromosomes.

This lets us compute, for every bin:
  - expected_depth(gc) via piecewise linear interpolation
  - variance(gc) estimated from the residuals in each GC bin
"""
import logging
from typing import List, Tuple

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

N_GC_BINS = 20  # number of equal-width GC% bins for the model


def fit_gc_model(
    pileup_list: List[np.ndarray],
    gc_list: List[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool (gc, depth) observations across all chromosomes and fit the model.

    Parameters
    ----------
    pileup_list : list of 1-D arrays
        Per-chromosome dense pileup arrays (float32, one value per bin).
    gc_list : list of 1-D arrays
        Per-chromosome GC fraction arrays (float, NaN for bins with no reads).

    Returns
    -------
    gc_centers : shape (N_GC_BINS,)  – centre of each GC% bin
    mean_depth : shape (N_GC_BINS,)  – mean depth in that GC bin (NaN if empty)
    var_depth  : shape (N_GC_BINS,)  – variance of depth (NaN if < 2 observations)
    """
    # Pool non-NaN, non-zero observations
    all_gc = np.concatenate([g.ravel() for g in gc_list])
    all_dp = np.concatenate([p.ravel() for p in pileup_list])
    valid = ~np.isnan(all_gc) & (all_dp > 0)
    gc_v = all_gc[valid]
    dp_v = all_dp[valid]

    edges = np.linspace(0.0, 1.0, N_GC_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_depth = np.full(N_GC_BINS, np.nan)
    var_depth = np.full(N_GC_BINS, np.nan)

    for i in range(N_GC_BINS):
        lo, hi = edges[i], edges[i + 1]
        if i == N_GC_BINS - 1:
            mask = (gc_v >= lo) & (gc_v <= hi)
        else:
            mask = (gc_v >= lo) & (gc_v < hi)
        dp_bin = dp_v[mask]
        if len(dp_bin) >= 2:
            mean_depth[i] = float(np.mean(dp_bin))
            var_depth[i] = float(np.var(dp_bin, ddof=1))
        elif len(dp_bin) == 1:
            mean_depth[i] = float(dp_bin[0])
            var_depth[i] = float(dp_bin[0])  # Poisson fallback

    n_valid = int(np.sum(~np.isnan(mean_depth)))
    logger.info(f'GC model fitted: {n_valid}/{N_GC_BINS} GC bins have data')
    return centers, mean_depth, var_depth


def predict_gc(
    gc_per_bin: np.ndarray,
    gc_centers: np.ndarray,
    mean_depth: np.ndarray,
    var_depth: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict expected depth and variance for each bin given its GC%.

    Bins with NaN GC% (no reads) get the global mean/variance as fallback.

    Returns
    -------
    expected : shape (n_bins,)  – GC-corrected expected depth per bin
    variance : shape (n_bins,)  – estimated variance per bin
    """
    valid_model = ~np.isnan(mean_depth)

    # Fallback values (global mean/var over all GC bins)
    global_mean = float(np.nanmean(mean_depth)) if np.any(valid_model) else 1.0
    global_var = float(np.nanmean(var_depth)) if np.any(valid_model) else 1.0

    n_bins = len(gc_per_bin)
    expected = np.full(n_bins, global_mean, dtype=np.float64)
    variance = np.full(n_bins, global_var, dtype=np.float64)

    if valid_model.sum() < 2:
        logger.warning('Too few GC model points; using global mean for Z-score correction')
        return expected, variance

    f_mean = interp1d(
        gc_centers[valid_model], mean_depth[valid_model],
        kind='linear', bounds_error=False,
        fill_value=(mean_depth[valid_model][0], mean_depth[valid_model][-1]),
    )
    f_var = interp1d(
        gc_centers[valid_model], var_depth[valid_model],
        kind='linear', bounds_error=False,
        fill_value=(var_depth[valid_model][0], var_depth[valid_model][-1]),
    )

    has_gc = ~np.isnan(gc_per_bin)
    expected[has_gc] = np.maximum(f_mean(gc_per_bin[has_gc]), 1e-6)
    raw_var = f_var(gc_per_bin[has_gc])
    # Variance must be at least the expected value (Poisson minimum)
    variance[has_gc] = np.maximum(raw_var, expected[has_gc])

    return expected, variance
