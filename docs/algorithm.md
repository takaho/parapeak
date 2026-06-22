# parapeak — Algorithm Reference

This document describes the full peak-calling pipeline: data flow, statistical models,
and the design decisions behind each stage.  For parameter descriptions and quick-start
examples, see the main README.

---

## Pipeline overview

```
BAM files
   │
   ▼
Stage 1 ── Single-pass pileup construction ────── (sequential, I/O bound)
   │         ├── QC filters (MAPQ, flags)
   │         ├── Coordinate-based deduplication (--keep-dup)
   │         ├── Fragment interval (TLEN or fixed extension)
   │         └── Per-chromosome integer arrays
   │
   ▼
Stage 2 ── Per-chromosome p-values ─────────────── (parallel, CPU bound)
   │         ├── NB p-values  (global + local background)
   │         └── Z-score p-values  (GC-corrected)
   │
   ▼
Stage 3 ── Genome-wide BH correction ───────────── (sequential, sync point)
   │         ├── Pool all chromosomes
   │         ├── BH over non-zero bins
   │         └── Scale q-values to all-bin denominator
   │
   ▼
Stage 4 ── Peak region assembly ────────────────── (parallel, CPU bound)
             ├── Merge significant bins
             ├── Signal filters (--min-count, --min-fold)
             └── PeakRecord objects → output files
```

---

## Stage 1: Read pileup construction

### 1a. QC filters

Applied to every read in the order listed.  The first failing condition discards the read.

| Order | Filter | SAM flag / field | Notes |
|---|---|---|---|
| 1 | Unmapped | `0x4` | No reference position |
| 2 | Duplicate flag | `0x400` | Set by `samtools markdup` / Picard |
| 3 | Secondary alignment | `0x100` | Non-primary multi-mapper |
| 4 | Supplementary alignment | `0x800` | Chimeric / split-read secondary |
| 5 | QC fail | `0x200` | Instrument-reported failure |
| 6 | Low MAPQ | `MAPQ < --min-mapq` | Default 20; excludes multi-mappers |
| 7 | Read 2 | `0x80` | Skipped to avoid double-counting fragments |

### 1b. Coordinate-based deduplication (`--keep-dup`)

Many genome-editing and ATAC-seq pipelines omit the duplicate-marking step, so the
`0x400` flag is never set even when the library contains extensive PCR amplification.
In GUIDE-seq and CIRCLE-seq libraries, the same cut-site event can be amplified to
hundreds of identical reads, producing spurious peaks with fold enrichment in the
hundreds at a single locus.

MACS3 addresses this with its `--keep-dup 1` default, which discards all but one read
per 5′-position + strand combination regardless of the SAM flag.  parapeak applies the
same logic after all SAM-flag filters have passed:

```
dedup key = (chromosome, 5′ position, strand)

forward read → 5′ position = reference_start  (leftmost aligned base)
reverse read → 5′ position = reference_end    (rightmost aligned base)
```

A counter tracks how many reads have been seen for each key.  When the count would
exceed `--keep-dup`, the read is discarded and the `filtered_coord_duplicate` counter
is incremented.

**Why 5′ position rather than fragment interval?**  Two paired-end reads from the
same PCR colony may have slightly different TLEN values due to end-repair variation
or soft-clipping differences, making `(start, end)` dedup overly permissive.  The 5′
end of a read is the enzymatic cleavage point in cut-site assays and is the most
stable identifier of a unique sequencing event.  This matches MACS3's single-end
model for ATAC/GUIDE-seq data.

**When to disable**: set `--keep-dup 0` if the BAM has already been processed with
`samtools markdup` or Picard `MarkDuplicates`, which set the `0x400` flag (already
handled by filter 2 above), to avoid a second deduplication pass.

### 1c. Fragment interval determination

After deduplication, the genomic interval to be counted for each read is determined:

**Paired-end proper pair** (TLEN in [`--min-fragment`, `--max-fragment`], R1 only):

```
Start = reference_start of R1            (if R1 is forward)
End   = reference_start + TLEN           (actual insert, right-open)

Start = reference_end + TLEN             (if R1 is reverse, TLEN is negative)
End   = reference_end
```

This uses the actual insert size, important for assays where fragment length carries
biological information (NFR vs nucleosomal fragments in ATAC-seq, circle diameter in
CIRCLE-seq).

**Single-end or out-of-range TLEN**:

```
forward: [reference_start, reference_start + fragment_size)
reverse: [reference_end − fragment_size, reference_end)
```

`fragment_size` is estimated automatically as the median TLEN of the first 100,000
proper pairs in the treated BAM; if no paired reads are found, 200 bp is used.
Override with `--fragment-size`.

**Guidance for cut-site assays**: For GUIDE-seq and CIRCLE-seq, set
`--fragment-size 100`.  The TLEN reflects the insert between R1 and R2 across the
full circle/junction, but the biologically informative region is the ~100 bp window
centred on the 5′ cleavage site.  This matches MACS3's `--extsize 100` for these
assay types.

### 1d. Pileup array

All bins spanned by the fragment interval are incremented by 1 in a pre-allocated
`int32` numpy array of length `ceil(chrom_size / bin_size)` for each chromosome.

GC content is accumulated simultaneously: for each read, the GC fraction of the
read sequence is added to a per-bin accumulator at the read's mapping start bin.
After all reads are processed, per-bin GC is averaged.

Blacklisted bins are zeroed after the pass completes.

---

## Stage 2: Per-chromosome p-values (parallel)

Each chromosome is dispatched to a worker process.  Workers are independent and
share no state after the initial data is sent.

### 2a. Negative-binomial p-values

#### Background model

The scaled control pileup is floored by a pseudocount:

```
bg[i] = ctrl[i] × scale_factor + pseudocount / local_bins
```

`pseudocount / local_bins` is the per-bin contribution of `pseudocount` reads spread
evenly across the local window.  This prevents zero-control bins from producing
near-zero background lambdas that would make any treatment signal appear maximally
significant.

#### Local background floor in control mode

When a negative-control BAM is provided, positions where the control has zero reads
produce a local background near `pseudocount / local_bins ≈ 0.01`.  In those regions,
even a modest treatment pileup of 2–5 reads yields a Poisson p-value near 10⁻¹³,
creating large numbers of false-positive peaks in control-mode runs.

To prevent this, the local background lambda is taken as the maximum of two sliding
window means:

```python
local_mean_ctrl  = sliding_window_mean(bg,                     local_bins)
local_mean_treat = sliding_window_mean(observed + pseudo_per_bin, local_bins)
local_mean       = maximum(local_mean_ctrl, local_mean_treat)
```

This mirrors MACS3's `max(lambda_bg, lambda_local)` approach: the statistical test is
always performed against whichever background is more conservative (higher lambda).
At positions where the control has zero coverage, `local_mean_treat` provides a floor
equal to the treatment's own local density, which is the same lambda that would be
used in no-control mode.

In no-control mode (`ctrl == treat`, `scale_factor == 1.0`), the two terms are
numerically identical, so this change does not affect no-control runs.

#### Global NB parameters

```
global_mean = mean(bg[~blacklist])
global_var  = var(bg[~blacklist and bg ≤ 99th percentile], ddof=1)
global_var  = max(global_var, global_mean)   # enforce NB ≥ Poisson
```

The 99th-percentile trim prevents peak-level bins from inflating the background
dispersion estimate.  Clamping variance to at least the mean ensures the NB
is never sub-Poisson (which would give anticonservative p-values).

NB parameters from method of moments:

```
r = global_mean² / (global_var − global_mean)   [or 1e6 if var ≤ mean → Poisson]
p = r / (r + global_mean)
```

Survival function `P(X ≥ k | NB(r, p))` is evaluated with a numba-JIT log-PMF
recurrence (see `stats._nb_sf_array`):

```
log P(X=0) = r·log(p)
log P(X=k) = log P(X=k−1) + log(r+k−1) − log(k) + log(1−p)
CDF[k] = Σ exp(log P(X=j)) for j = 0..k
P(X ≥ k) = 1 − CDF[k−1]
```

#### Local Poisson p-values

```
p_local[i] = P(X ≥ observed[i] | Poisson(local_mean[i]))
           = poisson.sf(observed[i] − 1, local_mean[i])
```

Scipy's `poisson.sf` is used; results are clipped to [10⁻³⁰⁰, 1].

#### Combined NB p-value

```
p_NB[i] = max(p_global[i], p_local[i])
```

The **larger** (less significant) p-value is retained.  A bin is only significant
when it is enriched above both the genome-wide mean **and** the local neighbourhood.
This is the primary guard against calling peaks in locally enriched background
(e.g., repeat-dense regions, GC-biased regions).

### 2b. GC-corrected Z-score p-values

#### GC model fitting

Per-bin GC fractions (from Stage 1) are collected for all non-blacklisted bins.
Bins are grouped into 20 equal-width GC intervals.  For each interval, the mean and
variance of the treated pileup are computed.  A piecewise-linear interpolation
gives a function `f(GC) → (expected_depth, variance)`.

The model is normalised so that the total expected depth equals the total observed
depth (scale the curve by the ratio of sums).

#### Z-score test

```
E[i], V[i] = f(GC[i])            [GC-corrected expected depth and variance]
V[i]       = max(V[i], 1e−6)     [floor]
Z[i]       = (observed[i] − E[i]) / √V[i]
p_Z[i]     = 1 − Φ(Z[i])         [one-sided upper-tail normal]
```

The Z-score test captures enrichment that the NB model misses due to GC-composition
differences between peaks and background (e.g., promoter peaks in AT-rich genomes).

---

## Stage 3: Genome-wide Benjamini–Hochberg correction

### The multiple-testing reference set

The human genome at 10 bp resolution has ≈ 300 million bins.  Only bins with at least
one read overlapping them have non-trivial p-values; the rest have p = 1 by definition
(Poisson test with observed = 0 always gives p = 1).

Running BH over only the non-zero bins (typically 1–4 million at MiSeq depth) would
use a denominator of n_nonzero ≈ 1 M, treating the problem as if the genome had
1 million tests rather than 300 million.  This makes BH 75–300× too lenient: the
q-value for the k-th ranked bin would be computed as `p_k × (n_nonzero / k)` rather
than the correct `p_k × (n_total / k)`.

### Scaling fix

parapeak runs BH over the n_nonzero active bins (efficient: avoids sorting 300 M
values) and then scales up the resulting q-values:

```
q_scaled = q_BH × (n_total / n_nonzero)
q_scaled = min(q_scaled, 1.0)
```

This is mathematically equivalent to running BH over all n_total bins with p = 1
assigned to zero-coverage bins, because:

- Zero-coverage bins have p = 1 → they are always ranked below the active bins
- Their contribution to the BH denominator is purely additive: each adds 1/n to the
  scale of q-values for the significant bins
- Multiplying active-bin q-values by n_total/n_nonzero achieves this without
  materialising the full 300 M array

The scale factor is reported in the log:

```
BH: 1,231,078 active / 309,992,336 total bins (scale 251.8x)
```

### Dual-method requirement

BH is run independently for the NB method and the Z-score method.  A bin is
**significant** only when both q_NB < threshold **and** q_Z < threshold (AND logic).
This dual-method requirement substantially reduces the false-positive rate compared
to either test alone, because the two tests have partially independent failure modes:

- NB is sensitive to over-dispersed (PCR-amplified) backgrounds
- Z-score is sensitive to GC-composition shifts between peak and background

A genuine enrichment peak should pass both.

---

## Stage 4: Peak region assembly

### Significant-bin mask

For each chromosome:

```
sig[i] = (q_NB[i] < threshold) AND (q_Z[i] < threshold) AND (not blacklisted[i])
```

### Gap filling

Small gaps between significant bins are bridged by binary dilation with a
structuring element of width `2 × max_gap_bins + 1`:

```python
sig_merged = binary_dilation(sig, structure=ones(2*max_gap_bins+1)) & ~blacklist
```

This allows peaks with a narrow dip in the middle (e.g., from a single poorly-mapped
read being excluded) to be called as a single region rather than two adjacent peaks.

### Region assembly

Contiguous runs of True in `sig_merged` are collected.  Each run is a candidate
region.  Runs shorter than `min_length_bins` are discarded.

### Signal filters

For each candidate region the following filters are applied in order.

#### Summit pileup (`--min-count`)

```
summit_pileup = max(treatment[start_bin:end_bin])
if summit_pileup < min_count: discard
```

This is an absolute signal filter independent of statistics.  At MiSeq depth with
paired-end 150 bp reads and fragment_size = 100–200 bp, a pileup of 5 corresponds to
roughly 3–5 unique molecules.  Raising this threshold is the most direct way to
increase specificity at the cost of sensitivity.

#### Fold enrichment (`--min-fold`)

```
mean_treat        = mean(treatment[start_bin:end_bin])
mean_ctrl_scaled  = mean(control[start_bin:end_bin]) × scale_factor
fold              = (mean_treat + pseudocount) / (mean_ctrl_scaled + pseudocount)
if fold < min_fold: discard
```

In no-control mode, `control` is replaced with a flat array equal to the genome-wide
mean treatment depth for the fold-enrichment calculation only (the statistical tests
use the self-referential background described above).  This makes the fold enrichment
reflect enrichment above the genome baseline rather than always equal to 1.

---

## Scale factor (control normalisation)

When a control BAM is provided, treatment and control pileups are compared on an equal
per-read basis using a scale factor:

```
scale_factor = reads_used_treated / reads_used_control
```

The control pileup at each bin is multiplied by `scale_factor` before background
estimation, fold-enrichment calculation, and NB/Poisson tests.  This accounts for
differences in sequencing depth between the two libraries.

---

## Output score columns

| Column | Definition |
|---|---|
| `fold_enrichment` | Mean (treat + pc) / (scaled ctrl + pc) across peak bins |
| `-log10(p_NB)` | Minimum NB p-value across peak bins, negative log₁₀ |
| `-log10(p_Z)` | Minimum Z-score p-value across peak bins, negative log₁₀ |
| `-log10(q_NB)` | BH-corrected q-value for NB, minimum across peak bins |
| `-log10(q_Z)` | BH-corrected q-value for Z-score, minimum across peak bins |
| `score` (narrowPeak) | `min(−10 × log₁₀(min(q_NB, q_Z)), 1000)` (UCSC score field) |

---

## GC model details

### Why GC correction matters

At shallow sequencing depth the per-bin read count follows a compound Poisson
process.  The expected depth is proportional to local GC content because:

1. PCR amplification efficiency varies with GC (high-GC regions are under-amplified
   and low-GC regions are over-amplified in some library preps).
2. Tn5 insertion (ATAC-seq) and some nucleases have sequence-composition preferences.

Without GC correction, the Z-score test would call peaks in GC-rich regions simply
because they have lower-than-average depth (apparently enriched relative to the
GC-biased expectation), or miss real peaks in GC-poor regions.

### Reference-free GC estimation

Rather than reading per-bin GC content from a reference FASTA, parapeak infers it
from the mapped read sequences.  The GC fraction of each read is a reliable proxy
for the reference GC at its alignment position (within ~150 bp).  This eliminates
the need to ship or index a large reference file.

### Model fitting

```
intervals = [0–5%, 5–10%, …, 95–100%]   (20 bins)
for each interval:
    bins_in_interval = [i : GC[i] ∈ interval and not blacklisted[i]]
    mean_depth[interval] = mean(treatment[bins_in_interval])
    var_depth[interval]  = var(treatment[bins_in_interval])

interpolate: GC_fraction → (expected_depth, variance)
```

Sparse intervals (fewer than 10 bins) are extrapolated from the neighbours.
The curve is a piecewise-linear interpolation (`numpy.interp`) over the 20 centres.

### Normalisation

The expected-depth curve is scaled so that its integral over the genome equals the
total observed treatment read count.  This ensures that the Z-test is centred (no
global over- or under-estimation of expected depth).

---

## Parallelism model

```
Main process:
  Stage 1 (sequential): reads all BAM files, fills per-chrom arrays
  ┌─ ProcessPoolExecutor (single pool for Stages 2 and 4) ─────────────────┐
  │  Stage 2 (pool.submit per chrom): dispatches p-value workers           │
  │  Stage 3 (sequential, inside pool context): BH correction              │
  │  Stage 4 (pool.submit per chrom): dispatches peak-calling workers      │
  └────────────────────────────────────────────────────────────────────────┘
  Write output files

Worker processes:
  Each worker receives a copy of one chromosome's pileup arrays via pickle.
  Workers do not share memory and produce no side effects (pure functions).
  Results (p-value arrays / peak lists) are returned via the result channel.
```

**Single shared pool**: Stages 2 and 4 reuse the same `ProcessPoolExecutor` so
worker processes are spawned once.  A second spawn/teardown cycle (and the
associated Python import + numba JIT-cache load time per worker) is avoided.

**Numba thread cap**: each worker's internal thread count is set to
`floor(cpu_count / n_workers)` at startup.  Without this cap, `p` workers each
start `cpu_count` OpenMP threads for numba's `@njit(parallel=True)` loops,
totalling `p × cpu_count` threads on a machine with `cpu_count` physical cores.

Worker count is set by `-p / --threads`.  The practical upper bound is the number
of chromosomes (typically 25 for hg38), beyond which additional workers are idle.

Memory per worker ≈ 2 × n_bins × 4 bytes (two float32 arrays: treat and ctrl).
For hg38 at 10 bp resolution: 2 × 30 M × 4 ≈ 240 MB per worker.

Empirical scaling on BINDS122/S1.bam (46k reads, hg38, MiSeq depth):

| -p | Wall time | Speedup |
|---:|---:|---:|
| 1 | 57.5 s | 1.00× |
| 2 | 41.9 s | 1.37× |
| 4 | 41.2 s | 1.39× |

Scaling is limited by: (1) sequential stages (Stage 1 + Stage 3 = 16% of runtime,
Amdahl ceiling 2.72× at p=4); (2) IPC serialisation of ~27 GB of numpy arrays
across all chromosomes; (3) work imbalance (chr1 is ~20× larger than chr22).
See `docs/benchmark.md` for the full analysis.

---

## Comparison with MACS3

| Aspect | MACS3 | parapeak |
|---|---|---|
| Parallelism | Single-threaded | Multi-process per chromosome |
| BAM input | Sorted + indexed required | Single pass, no sort/index needed |
| Statistical model | Poisson | NB + GC-corrected Z-score (both required) |
| Background model | max(lambda_BG, lambda_1k, lambda_10k) | max(lambda_global_NB, lambda_local_Poisson) |
| Local window | d / 1 kb / 10 kb (max of three) | single `--local-window` (default 10 kb) |
| Duplicate handling | `--keep-dup` (coord-based, default 1) | `--keep-dup` (coord-based, default 1) |
| GC correction | Not built-in | Built-in, reference-free |
| Genome size | User-supplied `--gsize` | Inferred from BAM headers |
| BH denominator | All genome bins | Scaled from non-zero bins (equivalent) |
| Blacklist | Not built-in | Applied at pileup time |
| Broken BAM headers | Error | Auto-repaired |

The `max(lambda_local_Poisson, lambda_global_NB)` design in parapeak has the same
intent as MACS3's `max(lambda_BG, lambda_local)` but uses a single configurable
local window (`--local-window`, default 10 kb) rather than MACS3's trio of windows
(d, 1k, 10k).

**Measured concordance on BINDS122/S1.bam** (GUIDE-seq, hg38, no-control mode,
`--fragment-size 100 --local-window 10000 -q 0.05`):

| Metric | Value |
|---|---:|
| parapeak peaks | 291 |
| MACS3 peaks | 209 |
| Recall — MACS3 peaks recovered by parapeak | **73.7%** (154/209) |
| Precision — parapeak peaks overlapping MACS3 | **52.6%** (153/291) |
| F1 | **0.614** |
| Jaccard (bp) | **0.203** |

parapeak recovers ~74% of MACS3 peaks and calls ~40% more peaks overall.  The extra
peaks reflect the GC-corrected Z-score test detecting enrichments that the single
Poisson model misses, though some may be false positives at the default threshold.
Use `-q 0.01` or `--min-count 8` for a more conservative call set.

For the full benchmark analysis see `docs/benchmark.md`.
