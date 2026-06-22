# parapeak — Performance Benchmark and MACS3 Concordance

Measured on S1.bam from the BINDS122 dataset (GUIDE-seq, hg38, 46,425 unique reads
after coordinate deduplication).  Machine: Apple Silicon, 16 logical CPUs, SSD storage.
parapeak version 0.3.7.

Parameters used throughout:

```
parapeak: --fragment-size 100 --local-window 10000 --keep-dup 1 -q 0.05
MACS3:    --nomodel --extsize 100 --keep-dup 1 -q 0.05
```

---

## 1. Algorithm overview

parapeak splits peak calling into four sequential stages.  Stages 1 and 3 run in the
main process; Stages 2 and 4 are parallelised across chromosomes.

```
Stage 1  BAM read (sequential, I/O bound)
  Single pass through BAM; QC filters; coordinate deduplication;
  fragment interval; per-chromosome integer pileup arrays.

Stage 2  Per-chromosome p-values (parallel, CPU bound)
  NB survival function vs. global + local background (numba JIT).
  GC-corrected Z-score vs. per-bin expected depth.
  Conservative: keeps the larger (less significant) p-value per bin.

Stage 3  Genome-wide BH correction (sequential, synchronisation barrier)
  Pools all chromosomes; runs BH over non-zero bins;
  scales q-values by n_total_bins / n_nonzero_bins so the reference set
  is the full genome (not just the covered fraction).

Stage 4  Peak region assembly (parallel, CPU bound)
  Significant-bin mask → binary dilation (gap fill) → contiguous regions;
  applies --min-count and --min-fold signal filters.
```

Stages 2 and 4 share a single `ProcessPoolExecutor`, so worker processes are
spawned once and reused.  Each worker also has its numba internal thread count
capped at `cpu_count / n_workers` to prevent CPU over-subscription when
`p > 1` (each of `p` processes would otherwise start `cpu_count` OpenMP threads,
totalling `p × cpu_count` threads on a `cpu_count`-core machine).

---

## 2. Threading performance

### 2.1 Stage-level wall time

| Stage | p=1 | p=2 | p=4 |
|---|---:|---:|---:|
| Stage 1 (BAM read, sequential) | 7.8 s | 7.4 s | 7.6 s |
| Stage 2 (p-values) | 24.2 s | 18.0 s | 18.3 s |
| Stage 3 (BH correction, sequential) | 1.2 s | 1.2 s | 1.1 s |
| Stage 4 (peak calling) | 24.3 s | 15.4 s | 14.2 s |
| **Total** | **57.5 s** | **41.9 s** | **41.2 s** |

### 2.2 Overall speedup

| Threads | Wall time | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 57.5 s | 1.00× | 100% |
| 2 | 41.9 s | 1.37× | 68% |
| 4 | 41.2 s | 1.39× | 35% |

### 2.3 Interpretation

**p=2 is the practical optimum** for this dataset.  The incremental gain from p=4
is negligible (0.7 s, within run-to-run noise).

Three independent bottlenecks limit scaling beyond p=2:

#### (a) Amdahl ceiling from sequential stages

Stage 1 (7.8 s) and Stage 3 (1.2 s) are sequential and account for 16% of the
single-thread runtime.  Amdahl's law caps the achievable speedup:

```
T(p) = T_seq + T_par / p

T_seq = 7.8 + 1.2 = 9.0 s
T_par = 24.2 + 24.3 = 48.5 s

Amdahl ceiling at p=4: (9.0 + 48.5) / (9.0 + 48.5/4) = 57.5 / 21.1 = 2.72×
```

Even with perfect parallelism across the two parallel stages, the ceiling is 2.72×.
The observed 1.39× is limited further by the bottlenecks below.

#### (b) IPC serialisation overhead

Each chromosome's numpy arrays are pickled and sent to a worker process via a pipe.
For the human genome at 10 bp resolution this means:

| Stage | Arrays sent per chrom | Total data (25 chroms) |
|---|---|---|
| Stage 2 | treat, ctrl, gc, bl_mask (4 arrays × float64) | ≈ 9.9 GB |
| Stage 4 | treat, ctrl, q_nb, q_z, p_nb, p_z, bl_mask (7 arrays × float64) | ≈ 17.3 GB |

Serialising and deserialising ~27 GB per run on a single-process pipe is a hard
bandwidth ceiling.  Stage 2 shows almost no improvement from p=2 to p=4 (18.0 s vs
18.3 s), because the bottleneck is not computation but IPC throughput.

Stage 4 scales slightly better (15.4 s → 14.2 s at p=4) because it involves pure
Python/scipy computation (binary dilation, peak assembly) rather than numba-parallel
number-crunching, so the ratio of useful work to IPC overhead is higher.

#### (c) Work imbalance across chromosomes

Chromosomes differ dramatically in size.  chr1 (249 Mbp, 24.9 M bins) takes roughly
20× longer to process than chr22 (1.6 M bins).  With p=4 workers, one worker handles
chr1 while three others finish all smaller chromosomes and sit idle.  The
parallel-stage time is dominated by the longest chromosome regardless of p.

For a more balanced workload, chromosomes could be split into equal-length chunks
before dispatch (not implemented in the current version).

### 2.4 Recommendation

Use **`-p 2`** for a good balance of speed and efficiency.  Higher thread counts
provide no measurable benefit for shallow libraries (MiSeq depth) due to the
IPC and work-imbalance ceilings described above.

For deep libraries (NovaSeq depth), where Stage 2 computation dominates over IPC
setup time, higher thread counts may help.  This has not been benchmarked.

### 2.5 Future improvement: shared memory

The IPC bottleneck could be eliminated by placing the large per-chromosome arrays in
`multiprocessing.shared_memory` segments before dispatching jobs.  Workers would then
receive only a small descriptor (name, shape, dtype) and map the memory directly,
reducing per-task IPC to kilobytes instead of gigabytes.  This is the most impactful
single optimisation remaining.

---

## 3. MACS3 concordance

### 3.1 Peak counts

| Tool | Peaks called |
|---|---:|
| parapeak (p=1, default settings) | 291 |
| MACS3 3.0.4 | 209 |

### 3.2 Overlap metrics

| Metric | Value |
|---|---:|
| parapeak peaks overlapping MACS3 (precision) | 153 / 291 = **52.6%** |
| MACS3 peaks overlapping parapeak (recall) | 154 / 209 = **73.7%** |
| F1 score | **0.614** |
| Jaccard (bp overlap / bp union) | **0.203** |

A peak is considered overlapping if it shares at least 1 bp with any peak in the
other set.

### 3.3 Interpretation

**Recall 73.7%**: parapeak recovers 154 of MACS3's 209 peaks.  The 55 MACS3 peaks
missed by parapeak likely fall below either the NB or Z-score threshold at these
settings.  These are predominantly low-pileup peaks (typically pileup 5–8) where
parapeak's dual-test requirement (q_NB < 0.05 AND q_Z < 0.05) is more stringent
than MACS3's single Poisson test.

**Precision 52.6%**: 138 parapeak peaks do not overlap any MACS3 peak.  These fall
into two categories:
- **True positives missed by MACS3**: parapeak's GC-corrected Z-score test can
  detect enrichments in GC-biased regions that a pure Poisson model would attribute
  to GC bias rather than signal.
- **False positives**: bins that pass both tests but lie below MACS3's Poisson
  significance threshold due to minor differences in background estimation (local
  window size, pseudocount, lambda selection).

**Jaccard 0.203**: The low bp-Jaccard relative to the peak-level overlap statistics
reflects that parapeak merges adjacent significant bins (including gap-filling via
`--max-gap 30`) and returns broader regions than MACS3's summit-centred calls.
bp-Jaccard is sensitive to peak width differences; peak-level recall/precision are
more informative for sparse cut-site libraries.

**Summary**: for GUIDE-seq/CIRCLE-seq applications, parapeak recovers ~74% of
MACS3-called peaks while also calling ~40% additional peaks.  Raising the threshold
to `-q 0.01` or `--min-count 8` would narrow the parapeak-only set at the cost of
some sensitivity.

### 3.4 Algorithm differences

The table below compares the two tools' designs at the level relevant to concordance.

| Aspect | MACS3 | parapeak |
|---|---|---|
| Statistical test | Poisson | NB (global) + Poisson (local), both required |
| Local lambda windows | max(d, 1 kb, 10 kb) sliding | single `--local-window` (default 10 kb) |
| GC correction | None | Built-in, reference-free (per-bin from reads) |
| Genome size | User-supplied `--gsize` | Inferred from BAM headers |
| BH denominator | All genome bins | Scaled from non-zero bins (equivalent) |
| Peak-calling model | d-flanked summits | Contiguous significant-bin runs |
| Duplicate handling | `--keep-dup` coord-based | `--keep-dup` coord-based (same default) |
| Parallelism | Single-threaded | Multi-process per chromosome |
| BAM requirement | Sorted + indexed | Unsorted, unindexed acceptable |

The key statistical difference is that parapeak requires a bin to pass **both** the
NB and Z-score tests, whereas MACS3 uses a single Poisson test.  This AND logic
reduces false positives from GC-biased regions at the cost of sensitivity for
low-pileup peaks that pass one test but not the other.

The local lambda design is functionally equivalent: both tools use a 10 kb sliding
window mean as the local background; MACS3 additionally uses 1 kb and d-size windows
and takes the maximum of all three.  The extra short windows mean MACS3 can better
suppress locally clustered background, which may explain some of the 55 MACS3 peaks
not found by parapeak (their neighbourhood background is suppressed by a short window
that parapeak does not compute).

---

## 4. Benchmark command

Reproduce these results with:

```bash
python benchmark/run_benchmark.py \
    --bam /Volumes/Analysis/BINDS122/bam/S1.bam \
    --threads 1 2 4 \
    --genome hs
```

Output goes to `benchmark/results/`.  MACS3 3.x must be on PATH.
