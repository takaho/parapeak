# parapeak

An NGS peak caller featuring parallel-by-chromosome computation, blacklist exclusion, and a dual statistical framework (negative binomial + GC-corrected Z-score).

**parapeak** is inspired by [MACS3](https://github.com/macs3-project/MACS) but differs in several key aspects:

| Feature | MACS3 | parapeak |
|---|---|---|
| Parallelism | Single-threaded | Multi-process, per chromosome |
| Genome size | User-supplied `--gsize` | Inferred from BAM headers |
| Statistical model | Poisson | Negative binomial **and** GC-corrected Z-score (both required) |
| GC correction | Separate tool | Built-in (no reference FASTA needed) |
| Blacklist | Not built-in | Applied at pileup construction time |
| Input | BAM/SAM/BED | BAM/SAM only |

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Dependencies: `pysam`, `numpy`, `scipy`, `numba`

> **numba** accelerates the negative-binomial survival function and the Benjamini–Hochberg isotonic step with JIT compilation.  
> If numba is unavailable, parapeak falls back transparently to scipy.

---

## Quick start

```bash
# Single-end NGS with input control and blacklist
parapeak -t ChIP.bam -c Input.bam \
         --blacklist hg38-blacklist.v2.bed.gz \
         -o results/ -n H3K27ac -p 8

# Multiple treatment BAMs, no control
parapeak -t rep1.bam rep2.bam -o results/ -p 4

# Adjust local background window and q-value threshold
parapeak -t ChIP.bam -c Input.bam \
         --local-window 5000 -q 0.01 -o results/
```

---

## Options

```
usage: parapeak [-h] -t BAM [BAM ...] [-c BAM [BAM ...]] [--blacklist BED]
                [-o DIR] [-n NAME] [--local-window BP] [--fragment-size BP]
                [--bin-size BP] [-q FLOAT] [--min-length BP] [--max-gap BP]
                [-p INT]

Input:
  -t, --treated BAM [BAM ...]   Treated BAM/SAM files (BAM index required)
  -c, --control BAM [BAM ...]   Control/input BAM/SAM files
  --blacklist BED               Blacklist BED file (plain or gzip-compressed)

Output:
  -o, --output DIR              Output directory (default: parapeak_output)
  -n, --name NAME               Output file prefix (default: parapeak)

Algorithm parameters:
  --local-window BP             Local background window size in bp (default: 1000)
  --fragment-size BP            Read extension length; estimated from paired-end
                                insert size if omitted, falls back to 200 bp
  --bin-size BP                 Pileup bin size in bp (default: 10)
  -q, --qvalue FLOAT            Q-value threshold for both methods (default: 0.05)
  --min-length BP               Minimum peak length in bp (default: 200)
  --max-gap BP                  Maximum gap between significant bins to merge (default: 30)
  -p, --threads INT             Number of parallel worker processes (default: 1)
```

---

## Output files

| File | Format | Description |
|---|---|---|
| `<name>_peaks.narrowPeak` | ENCODE narrowPeak (BED6+4) | Final peak calls |
| `<name>_summits.bed` | BED | Single-base summit positions |
| `<name>_peaks.tsv` | TSV | All score columns for downstream analysis |

The narrowPeak columns follow the [ENCODE specification](https://genome.ucsc.edu/FAQ/FAQformat.html#format12):
`chrom start end name score strand signalValue -log10(p) -log10(q) summit_offset`

---

## Algorithm

### 1. Read pileup

Each BAM/SAM file is read with [pysam](https://pysam.readthedocs.io/).
Chromosomes are processed in parallel (`-p` workers). Within each chromosome,
reads are extended from their 5′ end by `--fragment-size` bp and accumulated
into a fixed-width bin array (`--bin-size`, default 10 bp).
Blacklisted bins are zeroed immediately after construction.

Fragment size is estimated automatically from paired-end insert sizes
(median of the first 100,000 proper pairs). Single-end libraries fall
back to 200 bp unless `--fragment-size` is specified.

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

### 3. Negative-binomial p-values

For each bin the background read count is estimated at two scales:

- **Global λ**: genome-wide mean of the scaled control pileup
  (or treatment pileup when no control is provided).
- **Local λ**: sliding-window mean of the scaled control pileup over
  `--local-window` bp centred on the bin.

The **larger** of the two lambda values is used as the background expectation,
making the test conservative: enrichment is only called when the signal
exceeds both the local and global baselines.

A Negative Binomial distribution NB(*r*, *p*) is fitted to the global
background by the method of moments:

```
r = μ² / (σ² − μ)     p = μ / σ²
```

where μ and σ² are the mean and variance of non-blacklisted control bins.
When σ² ≤ μ (underdispersion), the distribution degenerates to Poisson
(*r* → ∞).

The p-value for each bin is the upper-tail probability:

```
p_NB = P(X ≥ observed | NB(r, p))
```

The NB survival function is evaluated with a numba-JIT recurrence over
the log-PMF, enabling parallel computation across all bins.

### 4. GC-corrected Z-score p-values

Using the GC correction model from Step 2, each bin receives a
GC-matched expected depth *E* and variance *V*:

```
Z = (observed − E) / √V
p_Z = P(Z_standard > Z)   [one-sided upper tail]
```

The expected values are scaled so that their genome-wide sum matches the
total treatment read count, preserving library-size normalization.

### 5. Genome-wide Benjamini–Hochberg correction

After p-values are computed for all chromosomes, the two p-value vectors
are concatenated genome-wide and corrected independently:

```
q_NB = BH(p_NB)
q_Z  = BH(p_Z)
```

The BH isotonic enforcement step (cumulative minimum from the right) is
JIT-compiled with numba for speed.

A bin is declared **significant** only when **both** q-values fall below
the threshold (`-q`, default 0.05).

### 6. Peak region assembly

Contiguous runs of significant bins are merged across gaps of up to
`--max-gap` bp. Regions shorter than `--min-length` bp are discarded.

The summit of each peak is the bin with the highest treatment signal.
Fold enrichment is computed as:

```
fold_enrichment = mean_treatment / max(mean_scaled_control, ε)
```

Scores reported in the output are −log₁₀ of the minimum q-value across
the two methods.

---

## Comparison with MACS3

MACS3 uses a linked-list data structure internally, which limits its
ability to parallelise across chromosomes. parapeak replaces this with
per-chromosome NumPy arrays, making straightforward use of Python
`multiprocessing` with essentially no inter-process communication until
the genome-wide BH correction step.

The true runtime bottleneck is BAM decompression (handled by the C library
underlying pysam) and gzip decompression of the blacklist. These I/O costs
dominate over Python-level arithmetic, so there is no need for a compiled
extension language; numba JIT is sufficient to accelerate the remaining
compute-intensive statistics.

---

## Author

Takaho A. Endo

## License

MIT
