# parapeak

An NGS peak caller supporting shallow (MiSeq) to deep coverage across
ATAC-seq, genome-editing assays (CIRCLE-seq, GUIDE-seq), ChIP-seq, CUT&TAG,
and similar experiments.

**parapeak** is inspired by [MACS3](https://github.com/macs3-project/MACS) but differs in several key aspects:

| Feature | MACS3 | parapeak |
|---|---|---|
| Parallelism | Single-threaded | Multi-process, per chromosome |
| Genome size | User-supplied `--gsize` | Inferred from BAM headers |
| Statistical model | Poisson | Negative binomial **and** GC-corrected Z-score (both required) |
| GC correction | Separate tool | Built-in (no reference FASTA needed) |
| Blacklist | Not built-in | Applied at pileup construction time |
| Low-coverage support | Limited | Signal filters for MiSeq-depth data |
| Paired-end fragments | Fixed extension | Actual insert size from TLEN field |
| Input | BAM/SAM/BED | BAM/SAM only |

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Dependencies: `pysam`, `numpy`, `scipy`, `numba`

> **numba** accelerates the negative-binomial survival function and the
> Benjamini–Hochberg isotonic step with JIT compilation.
> If numba is unavailable, parapeak falls back transparently to scipy.

---

## Quick start

```bash
# Paired-end ATAC-seq (deep)
parapeak -t atac.bam --blacklist hg38-blacklist.v2.bed.gz \
         -o results/ -n ATAC -p 8

# CIRCLE-seq / genome-editing with NTC control (MiSeq depth)
parapeak -t treated.bam -c ntc.bam \
         --min-count 3 --min-fold 3.0 --pseudocount 2.0 \
         -o results/ -n edit -p 8

# Multiple replicates, no control
parapeak -t rep1.bam rep2.bam -o results/ -p 4
```

---

## Options

```
Input:
  -t, --treated BAM [BAM ...]   Treated BAM/SAM files (BAM index required)
  -c, --control BAM [BAM ...]   Control/input BAM/SAM files
  --blacklist BED               Blacklist BED file (plain or gzip-compressed)

Output:
  -o, --output DIR              Output directory (default: parapeak_output)
  -n, --name NAME               Output file prefix (default: parapeak)

Algorithm parameters:
  --local-window BP             Local background window size in bp (default: 1000)
  --fragment-size BP            Fallback extension for single-end or discordant
                                paired-end reads; estimated from paired-end insert
                                sizes if omitted, falls back to 200 bp
  --bin-size BP                 Pileup bin size in bp (default: 10)
  -q, --qvalue FLOAT            Q-value threshold for both methods (default: 0.05)
  --min-length BP               Minimum peak length in bp (default: 200)
  --max-gap BP                  Maximum gap to merge between significant bins (default: 30)

Signal filters:
  --min-count FLOAT             Minimum pileup at the peak summit bin (default: 5)
  --min-fold FLOAT              Minimum fold enrichment over background (default: 2.0)
  --pseudocount FLOAT           Background floor per local window (default: 1.0)

  -p, --threads INT             Number of parallel worker processes (default: 1)
```

---

## Single-end and paired-end handling

### Single-end

Each mapped read is extended from its 5′ end by `--fragment-size` bp toward
the sequenced fragment's expected end.  `--fragment-size` is estimated
automatically as the median insert size if the BAM contains paired reads;
otherwise it defaults to 200 bp.

```
5'─────read──────3'──────────────extension──────────────►
|←────────────── fragment_size ────────────────────────→|
```

### Paired-end

Only **R1** is processed; R2 is skipped to avoid double-counting fragments.

For **proper pairs** whose TLEN (template length from the SAM field) falls
within 10–2000 bp, the actual fragment interval is used directly:

```
R1  5'──────────►
                    ◄──────────5'  R2
|←───────── TLEN (actual insert size) ─────────────────→|
```

This is important for experiments where fragment size varies:

- **ATAC-seq**: each pair of Tn5 cut sites defines an accessible region;
  using actual insert sizes correctly captures nucleosome-free regions
  (NFR, < 200 bp) and mono-nucleosomal fragments (∼200 bp).
- **CIRCLE-seq / GUIDE-seq**: read pairs span the circularised DNA around
  cut sites; TLEN reflects the circle diameter.
- **ChIP-seq paired-end**: sonication produces a range of fragment sizes;
  actual TLEN avoids over-smoothing short fragments.

For **discordant pairs** (TLEN outside 10–2000 bp) and **unpaired reads**,
the read is treated as single-end and extended by `--fragment-size`.

---

## Signal filters for low-coverage data

At shallow sequencing depth (MiSeq, ~1–5 M reads), stochastic sampling
means that ~60% of 10 bp bins in the control have zero reads.
Without a signal floor, NB and Z-score tests can declare bins with only
1–2 reads significant wherever the control happens to have zero coverage,
producing thousands of artefactual peaks.

Three filters address this, all applied *after* the BH q-value test:

### `--pseudocount` (default: 1.0)

Adds a pseudocount (in units of reads per `--local-window`) to the scaled
control pileup before NB background fitting and fold-enrichment calculation.
Distributing the pseudocount across the local window gives each 10 bp bin
a minimum background floor of `pseudocount / (local_window / bin_size)`.

Fold enrichment is computed as:

```
fold = (mean_treatment + pseudocount) / (mean_scaled_control + pseudocount)
```

This eliminates infinite fold values when the control has zero reads and
makes the metric interpretable at all coverage depths.

### `--min-count` (default: 5.0)

Discards peaks whose summit bin has a pileup value below this threshold,
regardless of the q-value.
At typical fragment extension (200 bp), a summit pileup of 5 corresponds
to approximately 2–3 overlapping read fragments; for sparser data where
even 2–3 reads can be statistically significant against zero control, this
filter prevents low-signal artefacts.

### `--min-fold` (default: 2.0)

Discards peaks with fold enrichment (pseudocount-corrected) below this
value.  Combined with `--pseudocount`, a minimum fold of 2.0 ensures that
the signal is at least twice the effective background, not just twice zero.
Set to values < 1 to disable.

---

## Recommended parameters by experiment

| Experiment | SE/PE | Typical depth | `--min-count` | `--min-fold` | `--pseudocount` | `--local-window` | Notes |
|---|---|---|---|---|---|---|---|
| ATAC-seq (deep) | PE | > 20 M | 10 | 2.0 | 0.5 | 1000 | Nucleosome-free regions; actual insert sizes used |
| ATAC-seq (MiSeq) | PE | 2–5 M | 3 | 2.0 | 2.0 | 2000 | Wider local window compensates for sparse control |
| CIRCLE-seq | PE | 1–5 M | 3 | 3.0 | 2.0 | 1000 | Sparse library; higher fold reduces false positives |
| GUIDE-seq | PE | 5–20 M | 5 | 2.0 | 1.0 | 1000 | On-target site may need blacklisting |
| ChIP-seq narrow (deep) | SE/PE | > 10 M | 10 | 2.0 | 0.5 | 1000 | TF binding sites |
| ChIP-seq narrow (MiSeq) | SE | 1–5 M | 3 | 2.0 | 2.0 | 5000 | Wider local window for stable background |
| ChIP-seq broad | SE/PE | > 10 M | 5 | 1.5 | 1.0 | 5000 | Histone marks; lower fold threshold |
| CUT&TAG | PE | 5–20 M | 5 | 2.0 | 1.0 | 500 | High signal-to-noise; narrow local window |
| FAIRE-seq | SE/PE | > 5 M | 5 | 2.0 | 1.0 | 1000 | Similar to ATAC-seq |

### Coverage guide

| Sequencer | Typical read pairs | Recommended `--min-count` | Recommended `--pseudocount` |
|---|---|---|---|
| NovaSeq / HiSeq (deep) | > 30 M | 10–20 | 0.5 |
| NextSeq / HiSeq (standard) | 10–30 M | 5–10 | 1.0 |
| MiSeq (V3 2×300) | 3–10 M | 3–5 | 1.0–2.0 |
| MiSeq (V2 2×150) | 1–3 M | 3 | 2.0–5.0 |

---

## Output files

| File | Format | Description |
|---|---|---|
| `<name>_peaks.narrowPeak` | ENCODE narrowPeak (BED6+4) | Final peak calls |
| `<name>_summits.bed` | BED | Single-base summit positions |
| `<name>_peaks.tsv` | TSV | All score columns for downstream analysis |

The narrowPeak columns follow the
[ENCODE specification](https://genome.ucsc.edu/FAQ/FAQformat.html#format12):

```
chrom  start  end  name  score  strand  signalValue  -log10(p)  -log10(q)  summit_offset
```

The TSV file includes separate NB and Z-score p-values and q-values for
inspection of which statistical test is the driver for each peak.

---

## Algorithm

### 1. Read pileup

Each BAM/SAM file is read with [pysam](https://pysam.readthedocs.io/).
Chromosomes are processed in parallel (`-p` workers).
Fragment intervals are determined as described in the
[Single-end and paired-end handling](#single-end-and-paired-end-handling)
section.  Each fragment adds 1 to every 10 bp bin it overlaps.
Blacklisted bins are zeroed immediately after construction.

### 2. GC correction model

Without requiring a reference genome, parapeak estimates per-bin GC content
from the sequences of mapped reads (a reliable proxy for reference GC%).

The genome is divided into non-overlapping windows (`--local-window`).
For each window, the mean GC fraction of overlapping reads and the observed
read depth are recorded. All windows across all chromosomes are pooled,
binned into 20 equal GC% intervals (0–5%, 5–10%, …, 95–100%), and the
mean and variance of depth are computed per interval.

A piecewise-linear interpolation curve `f(GC%) → (expected_depth, variance)`
is fitted to these bin statistics. This curve is used in Step 4 to derive
GC-corrected expected values and their variances for every genomic bin.
When too few data points are available (sparse libraries), the model falls
back to the global mean depth.

### 3. Negative-binomial p-values

The scaled control pileup is floored by adding
`pseudocount / (local_window / bin_size)` per bin before fitting.

For each bin, the background is estimated at two scales:

- **Global λ**: genome-wide mean of the pseudocount-adjusted scaled control.
- **Local λ**: sliding-window mean over `--local-window` bp.

The **larger** lambda (more conservative, higher p-value) is used, so
enrichment is only called when signal exceeds both local and global baselines.

A Negative Binomial NB(*r*, *p*) is fitted by the method of moments:

```
r = μ² / (σ² − μ)     p = μ / σ²
```

When σ² ≤ μ the distribution degenerates to Poisson (*r* → ∞).
The survival function is evaluated with a numba-JIT log-PMF recurrence.

### 4. GC-corrected Z-score p-values

Each bin receives a GC-matched expected depth *E* and variance *V* from
Step 2.  Expected values are scaled to match the total treatment read count.

```
Z = (observed − E) / √V
p_Z = P(Z_standard > Z)   [one-sided upper tail]
```

### 5. Genome-wide Benjamini–Hochberg correction

P-values from all chromosomes are pooled and corrected independently for
each method:

```
q_NB = BH(p_NB)     q_Z = BH(p_Z)
```

A bin is **significant** only when **both** q_NB < threshold **and**
q_Z < threshold.

### 6. Peak region assembly and signal filtering

Contiguous significant bins are merged across gaps of up to `--max-gap` bp.
Regions shorter than `--min-length` bp are discarded.

Signal filters are then applied in order:

1. **`--min-count`**: summit pileup ≥ threshold
2. **`--min-fold`**: fold enrichment (pseudocount-corrected) ≥ threshold

The summit is the bin with the highest treatment pileup.
The score in the output is −log₁₀ of the minimum q-value across both methods.

---

## Comparison with MACS3

MACS3 uses a linked-list data structure internally, which limits its
ability to parallelise across chromosomes. parapeak replaces this with
per-chromosome NumPy arrays and Python `multiprocessing`, with
inter-process communication only at the genome-wide BH correction step.

The true runtime bottleneck is BAM decompression (via the C library
underlying pysam) and gzip decompression of the blacklist. These I/O costs
dominate over Python-level arithmetic, so there is no need for a compiled
extension language; numba JIT is sufficient for the compute-intensive
statistical routines.

---

## Author

Takaho A. Endo

## License

MIT
