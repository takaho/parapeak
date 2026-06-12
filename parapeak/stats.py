"""
Statistical tests and multiple-testing correction.

Hot paths (NB p-value loop, BH isotonic step) are JIT-compiled with numba
for speed.  All other code is plain NumPy/SciPy.

NB p-value
----------
For each bin we compute P(X >= observed | NB(r, p)) using both a
*global* background λ and a *local* background λ (sliding window).
We take the **larger** p-value (more conservative background) per bin.

Z-score p-value
---------------
GC-corrected expected depth and variance; one-sided upper-tail normal.

BH correction
-------------
Applied genome-wide to each method independently.
Peaks require BOTH q_nb < threshold AND q_z < threshold.
"""
import logging
from typing import Tuple

import numpy as np
from scipy import ndimage
from scipy.special import betaln
from scipy.stats import norm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# numba JIT helpers
# ---------------------------------------------------------------------------
try:
    from numba import njit, prange

    @njit(parallel=True, cache=True)
    def _nb_sf_array(observed: np.ndarray, r: float, p: float) -> np.ndarray:
        """
        Survival function of NB(r, p) evaluated at each element of *observed*.
        P(X >= k) = I_{1-p}(k, r)  where I is the regularised incomplete beta.

        We use the log-probability recurrence instead of the incomplete beta to
        keep things numba-friendly:
          log P(X=0) = r * log(p)
          log P(X=k) = log P(X=k-1) + log(r+k-1) - log(k) + log(1-p)
        Accumulate from 0 up to max(observed), then sum the tail.

        For speed the recurrence is computed once up to max_k, then each output
        is filled in a parallel loop.
        """
        n = len(observed)
        if n == 0:
            return np.empty(0, dtype=np.float64)

        max_k = int(np.max(observed))
        if max_k < 0:
            return np.ones(n, dtype=np.float64)

        # Build CDF
        log_pmf = np.empty(max_k + 1, dtype=np.float64)
        log_p = np.log(max(p, 1e-300))
        log_1mp = np.log(max(1.0 - p, 1e-300))
        log_pmf[0] = r * log_p
        for k in range(1, max_k + 1):
            log_pmf[k] = log_pmf[k - 1] + np.log(r + k - 1) - np.log(k) + log_1mp

        # CDF (inclusive)
        cdf = np.empty(max_k + 1, dtype=np.float64)
        cdf[0] = np.exp(log_pmf[0])
        for k in range(1, max_k + 1):
            cdf[k] = cdf[k - 1] + np.exp(log_pmf[k])
        cdf = np.minimum(cdf, 1.0)

        out = np.empty(n, dtype=np.float64)
        for i in prange(n):
            k = int(observed[i])
            if k <= 0:
                out[i] = 1.0
            elif k - 1 >= max_k:
                out[i] = 1.0 - cdf[max_k]
            else:
                out[i] = 1.0 - cdf[k - 1]
        return out

    @njit(cache=True)
    def _bh_isotonic(q_sorted: np.ndarray) -> np.ndarray:
        """Enforce non-increasing sequence from the right (cumulative minimum)."""
        n = len(q_sorted)
        result = q_sorted.copy()
        min_val = 1.0
        for i in range(n - 1, -1, -1):
            if result[i] > min_val:
                result[i] = min_val
            else:
                min_val = result[i]
        return result

    _NUMBA_AVAILABLE = True
    logger.debug('numba JIT compilation enabled for stats hot paths')

except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False
    logger.info('numba not available; falling back to scipy for NB survival function')


# ---------------------------------------------------------------------------
# NB parameter fitting
# ---------------------------------------------------------------------------

def _nb_params(mean: float, var: float) -> Tuple[float, float]:
    """
    Fit NB parameters (r, p) from empirical (mean, variance).

      r = mean² / (var − mean),   p = mean / var

    Falls back to near-Poisson NB (large r) when var ≤ mean.
    """
    if mean <= 0:
        return 1e6, 1e6 / (1e6 + 1e-9)
    if var <= mean:
        r = 1e6
    else:
        r = mean ** 2 / (var - mean)
    p = r / (r + mean)
    return float(r), float(p)


def _nb_sf(observed: np.ndarray, r: float, p: float) -> np.ndarray:
    """P(X >= observed) under NB(r, p)."""
    if _NUMBA_AVAILABLE:
        return _nb_sf_array(observed.astype(np.float64), r, p)
    # scipy fallback
    from scipy.stats import nbinom
    obs_int = np.maximum(observed, 0).astype(np.int32)
    return nbinom.sf(obs_int - 1, r, p)


# ---------------------------------------------------------------------------
# Local background (sliding window)
# ---------------------------------------------------------------------------

def compute_local_background(pileup: np.ndarray, local_bins: int) -> np.ndarray:
    """Sliding-window mean with reflect boundary."""
    return ndimage.uniform_filter1d(
        pileup.astype(np.float64), size=max(1, local_bins), mode='reflect'
    )


# ---------------------------------------------------------------------------
# NB p-values
# ---------------------------------------------------------------------------

def nb_pvalues(
    observed: np.ndarray,
    ctrl_pileup: np.ndarray,
    local_bins: int,
    scale_factor: float,
    blacklist_mask: np.ndarray,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """
    Compute per-bin NB p-values.

    Background is the scaled control signal + pseudocount, evaluated at
    both global and local levels.  The **larger** p-value (more conservative,
    less significant) is returned per bin.

    Pseudocount
    -----------
    ``pseudocount`` (in units of reads per local window) is spread evenly
    across the ``local_bins`` bins of each window and added to the scaled
    control before fitting the NB distribution.  This floors the background
    at a minimum level even when the control has zero reads at a position,
    preventing stochastic zero-count bins from producing artificially tiny
    p-values at low coverage.

    At typical MiSeq depth the control pileup is zero for ~60 % of bins.
    Without a pseudocount those bins produce P(X ≥ k | λ_global) which
    can pass BH correction even for moderate k, generating a flood of
    false-positive peaks on any chromosome that happens to be
    under-represented in the control library.
    """
    pseudo_per_bin = pseudocount / max(local_bins, 1)
    bg = ctrl_pileup * scale_factor + pseudo_per_bin
    valid = ~blacklist_mask

    global_mean = float(np.mean(bg[valid])) if valid.any() else float(np.mean(bg))
    global_var  = float(np.var(bg[valid], ddof=1)) if valid.sum() > 1 else global_mean
    global_var  = max(global_var, global_mean)

    local_mean = compute_local_background(bg, local_bins)

    # Global background p-values
    r_g, p_g = _nb_params(global_mean, global_var)
    pvals_global = _nb_sf(observed, r_g, p_g)

    # Local background p-values (same dispersion, local lambda)
    local_global_mean = float(np.mean(local_mean[valid])) if valid.any() else global_mean
    r_l, p_l = _nb_params(local_global_mean, global_var)
    pvals_local = _nb_sf(observed, r_l, p_l)

    # Conservative: take the larger p-value (less significant)
    pvals = np.maximum(pvals_global, pvals_local)
    return np.clip(pvals, 1e-300, 1.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Z-score p-values (GC-corrected)
# ---------------------------------------------------------------------------

def zscore_pvalues(
    observed: np.ndarray,
    expected: np.ndarray,
    variance: np.ndarray,
) -> np.ndarray:
    """One-sided upper-tail p-value from GC-corrected Z-score."""
    variance = np.maximum(variance, 1e-6)
    z = (observed.astype(np.float64) - expected) / np.sqrt(variance)
    return np.clip(norm.sf(z), 1e-300, 1.0)


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg correction returning q-values in input order.
    The isotonic (cumulative-minimum) step is JIT-compiled when numba is
    available.
    """
    n = len(pvalues)
    if n == 0:
        return np.array([], dtype=np.float64)

    order = np.argsort(pvalues)
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)
    q = pvalues * n / ranks

    q_sorted = q[order]
    if _NUMBA_AVAILABLE:
        q_sorted = _bh_isotonic(q_sorted)
    else:
        min_val = 1.0
        for i in range(n - 1, -1, -1):
            if q_sorted[i] > min_val:
                q_sorted[i] = min_val
            else:
                min_val = q_sorted[i]

    q[order] = q_sorted
    return np.minimum(q, 1.0)
