#!/usr/bin/env python3
"""
Benchmark parapeak: threading speedup and MACS3 concordance.

Runs S1.bam (no control) at -p 1, 2, 4 threads; extracts per-stage
timings from log output; then runs MACS3 with equivalent parameters
and reports peak-call concordance.

Usage
-----
    python benchmark/run_benchmark.py [options]

    --bam PATH        BAM file to benchmark (default: BINDS122/S1.bam)
    --threads N ...   Thread counts to test (default: 1 2 4)
    --repeats N       Repeats per thread count for stable timing (default: 1)
    --skip-macs3      Skip MACS3 concordance comparison
    --genome STR      MACS3 genome size key (default: hs)
    --output DIR      Directory for run outputs (default: benchmark/results)
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).parent.parent
DEFAULT_BAM = '/Volumes/Analysis/BINDS122/bam/S1.bam'
DEFAULT_THREADS = [1, 2, 4]

# parapeak parameters equivalent to MACS3 --nomodel --extsize 100
PP_FIXED_ARGS = [
    '--fragment-size', '100',
    '--local-window',  '10000',
    '--keep-dup',      '1',
    '--bigwig-bin',    '0',    # skip bigWig to reduce I/O noise in timing
]


# ---------------------------------------------------------------------------
# Interval utilities (no external dependencies)
# ---------------------------------------------------------------------------

def _parse_narrowpeak(path: str) -> List[Tuple[str, int, int]]:
    peaks: List[Tuple[str, int, int]] = []
    with open(path) as fh:
        for line in fh:
            if line.startswith('#') or not line.strip():
                continue
            p = line.split('\t')
            if len(p) >= 3:
                peaks.append((p[0], int(p[1]), int(p[2])))
    peaks.sort()
    return peaks


def _parse_pp_narrowpeak(path: str) -> List[Tuple[str, int, int]]:
    return _parse_narrowpeak(path)


def _total_bp(peaks: List[Tuple[str, int, int]]) -> int:
    return sum(e - s for _, s, e in peaks)


def _bp_overlap(
    a: List[Tuple[str, int, int]],
    b: List[Tuple[str, int, int]],
) -> int:
    """Merge-sweep total base-pair overlap between two sorted interval lists."""
    ia = ib = total = 0
    while ia < len(a) and ib < len(b):
        ca, sa, ea = a[ia]
        cb, sb, eb = b[ib]
        if ca < cb or (ca == cb and ea <= sb):
            ia += 1
        elif cb < ca or (ca == cb and eb <= sa):
            ib += 1
        else:
            ov = min(ea, eb) - max(sa, sb)
            if ov > 0:
                total += ov
            if ea <= eb:
                ia += 1
            else:
                ib += 1
    return total


def _count_overlapping(
    query: List[Tuple[str, int, int]],
    ref:   List[Tuple[str, int, int]],
) -> int:
    """Count query peaks that have ≥1 bp overlap with any ref peak."""
    count = ri = 0
    for cq, sq, eq in query:
        while ri < len(ref) and (ref[ri][0] < cq or
              (ref[ri][0] == cq and ref[ri][2] <= sq)):
            ri += 1
        rj = ri
        while rj < len(ref) and ref[rj][0] == cq and ref[rj][1] < eq:
            if ref[rj][2] > sq:
                count += 1
                break
            rj += 1
    return count


def concordance(
    pp: List[Tuple[str, int, int]],
    m3: List[Tuple[str, int, int]],
) -> Dict:
    pp_bp  = _total_bp(pp)
    m3_bp  = _total_bp(m3)
    ov_bp  = _bp_overlap(pp, m3)
    union  = pp_bp + m3_bp - ov_bp

    pp_ov  = _count_overlapping(pp, m3)
    m3_ov  = _count_overlapping(m3, pp)

    prec   = pp_ov / max(len(pp), 1)
    rec    = m3_ov / max(len(m3), 1)
    f1     = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        'n_parapeak':   len(pp),
        'n_macs3':      len(m3),
        'pp_overlap':   pp_ov,
        'm3_overlap':   m3_ov,
        'precision':    prec,
        'recall':       rec,
        'f1':           f1,
        'jaccard_bp':   ov_bp / max(union, 1),
    }


# ---------------------------------------------------------------------------
# Stage timing parser
# ---------------------------------------------------------------------------

_STAGE_RE = re.compile(
    r'Stage (\d+) \(([^)]+)\):\s*([\d.]+)\s*s'
)
_TOTAL_RE = re.compile(r'Total pipeline:\s*([\d.]+)\s*s')


def parse_stage_times(log: str) -> Dict[str, float]:
    times: Dict[str, float] = {}
    for m in _STAGE_RE.finditer(log):
        times[f'Stage {m.group(1)} ({m.group(2)})'] = float(m.group(3))
    m = _TOTAL_RE.search(log)
    if m:
        times['Total pipeline'] = float(m.group(1))
    return times


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_parapeak(
    bam: str,
    outdir: str,
    name: str,
    threads: int,
    extra_args: Optional[List[str]] = None,
) -> Tuple[float, str]:
    """Return (wall_seconds, stderr_log)."""
    cmd = [
        sys.executable, '-m', 'parapeak',
        '-t', bam,
        '-o', outdir,
        '-n', name,
        '-p', str(threads),
    ] + PP_FIXED_ARGS + (extra_args or [])

    env = os.environ.copy()
    env['PYTHONPATH'] = str(REPO_ROOT)

    t0 = time.perf_counter()
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        cwd=str(REPO_ROOT), env=env,
    )
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        print(f'  [ERROR] parapeak exit {result.returncode}')
        print(result.stderr[-3000:])

    return wall, result.stderr


def run_macs3(
    bam: str,
    outdir: str,
    name: str,
    genome: str = 'hs',
) -> Tuple[float, str]:
    """Return (wall_seconds, stderr)."""
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        'macs3', 'callpeak',
        '-t', bam,
        '-f', 'BAM',
        '-n', name,
        '--outdir', outdir,
        '-g', genome,
        '--nomodel',
        '--extsize', '100',
        '--keep-dup', '1',
        '-q', '0.05',
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        print(f'  [ERROR] macs3 exit {result.returncode}')
        print(result.stderr[-3000:])

    return wall, result.stderr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--bam',        default=DEFAULT_BAM,     metavar='PATH')
    ap.add_argument('--threads',    nargs='+', type=int,     default=DEFAULT_THREADS)
    ap.add_argument('--repeats',    type=int,  default=1,    metavar='N')
    ap.add_argument('--skip-macs3', action='store_true')
    ap.add_argument('--genome',     default='hs',            metavar='STR')
    ap.add_argument('--output',     default=str(Path(__file__).parent / 'results'), metavar='DIR')
    args = ap.parse_args()

    result_dir = Path(args.output)
    result_dir.mkdir(parents=True, exist_ok=True)

    print(f'\n{"="*64}')
    print(f'  parapeak threading benchmark + MACS3 concordance')
    print(f'  BAM     : {args.bam}')
    print(f'  threads : {args.threads}')
    print(f'  repeats : {args.repeats}')
    print(f'{"="*64}\n')

    # -----------------------------------------------------------------------
    # Threading benchmark
    # -----------------------------------------------------------------------
    # wall_times[p] = list of wall-clock seconds (one per repeat)
    wall_times: Dict[int, List[float]] = {}
    # stage_times[p] = dict from stage label → seconds (last repeat)
    stage_times: Dict[int, Dict[str, float]] = {}
    last_outdir: Dict[int, str] = {}

    for p in args.threads:
        wall_times[p] = []
        for rep in range(args.repeats):
            name   = f'bench_p{p}_r{rep}'
            outdir = str(result_dir / name)
            shutil.rmtree(outdir, ignore_errors=True)

            suffix = f' (rep {rep+1}/{args.repeats})' if args.repeats > 1 else ''
            print(f'  parapeak -p {p}{suffix} ...', end='', flush=True)
            wall, log = run_parapeak(args.bam, outdir, name, p)
            wall_times[p].append(wall)
            print(f' {wall:.1f} s')

            if rep == args.repeats - 1:
                stage_times[p] = parse_stage_times(log)
                last_outdir[p] = outdir

    # Per-stage breakdown for all thread counts
    p_ref = args.threads[0]
    st = stage_times.get(p_ref, {})
    if st:
        print(f'\n{"─"*64}')
        print(f'  Per-stage timing:')
        # header
        hdr = f'  {"stage":<42}'
        for p in sorted(args.threads):
            hdr += f'  p={p:>2} (s)'
        print(hdr)
        print(f'  {"─"*42}' + '  ─────────' * len(args.threads))
        for label in sorted(st):
            row = f'  {label:<42}'
            for p in sorted(args.threads):
                v = stage_times.get(p, {}).get(label, float('nan'))
                if v == v:
                    row += f'  {v:>9.2f}'
                else:
                    row += f'  {"—":>9}'
            print(row)

    # Compute min wall time across repeats for stable comparison
    min_wall = {p: min(ws) for p, ws in wall_times.items()}
    base = min_wall[p_ref]

    print(f'\n{"─"*64}')
    print(f'  {"threads":>8}  {"best (s)":>9}  {"mean (s)":>9}  {"speedup":>8}  {"efficiency":>10}')
    print(f'  {"─"*8}  {"─"*9}  {"─"*9}  {"─"*8}  {"─"*10}')
    for p in sorted(args.threads):
        ws   = wall_times[p]
        best = min(ws)
        mean = sum(ws) / len(ws)
        spd  = base / best
        eff  = spd / p * 100
        print(f'  {p:>8}  {best:>9.1f}  {mean:>9.1f}  {spd:>8.2f}×  {eff:>9.0f}%')

    if len(args.threads) > 1:
        print()
        print('  Note: stage 1 (BAM read) and stage 3 (BH) are sequential.')
        p_max = max(args.threads)
        seq = sum(v for k, v in st.items()
                  if 'BAM' in k or 'BH' in k or 'correction' in k.lower())
        par = sum(v for k, v in st.items()
                  if 'p-value' in k.lower() or 'peak calling' in k.lower())
        if seq > 0 and par > 0:
            t1    = seq + par                  # single-thread pipeline time
            t_inf = t1 / (seq + par / p_max)  # Amdahl: parallel time with p_max workers
            # speedup = T1 / T_p = t1 / (seq + par/p)
            amdahl_spd = t1 / (seq + par / p_max)
            print(f'  Amdahl ceiling at p={p_max}: {amdahl_spd:.2f}× '
                  f'(seq={seq:.1f}s / total={t1:.1f}s = {seq/t1:.0%} sequential)')

    # -----------------------------------------------------------------------
    # MACS3 concordance
    # -----------------------------------------------------------------------
    if not args.skip_macs3:
        if not shutil.which('macs3'):
            print('\n  macs3 not found on PATH; skipping concordance.')
        else:
            m3_dir = str(result_dir / 'macs3')
            print(f'\n{"─"*64}')
            print('  Running MACS3 ...', end='', flush=True)
            m3_wall, _ = run_macs3(args.bam, m3_dir, 'S1', args.genome)
            print(f' {m3_wall:.1f} s')

            pp_np = str(Path(last_outdir[p_ref]) / f'bench_p{p_ref}_r{args.repeats-1}_peaks.narrowPeak')
            m3_np = str(Path(m3_dir) / 'S1_peaks.narrowPeak')

            if os.path.exists(pp_np) and os.path.exists(m3_np):
                pp_peaks = _parse_pp_narrowpeak(pp_np)
                m3_peaks = _parse_narrowpeak(m3_np)
                stats    = concordance(pp_peaks, m3_peaks)

                print(f'\n{"─"*64}')
                print('  MACS3 concordance  (S1.bam, no-control, --fragment-size 100):')
                print(f'    parapeak peaks               : {stats["n_parapeak"]:>6,}')
                print(f'    MACS3 peaks                  : {stats["n_macs3"]:>6,}')
                print()
                print(f'    parapeak peaks overlapping   : {stats["pp_overlap"]:>6,} / {stats["n_parapeak"]:,}  '
                      f'({stats["precision"]:.1%} precision)')
                print(f'    MACS3 peaks overlapping      : {stats["m3_overlap"]:>6,} / {stats["n_macs3"]:,}  '
                      f'({stats["recall"]:.1%} recall)')
                print()
                print(f'    F1 score                     : {stats["f1"]:.3f}')
                print(f'    Jaccard (bp overlap / union) : {stats["jaccard_bp"]:.3f}')
                print()
                print('  Parameters used:')
                print('    parapeak : --fragment-size 100 --local-window 10000 --keep-dup 1 -q 0.05')
                print('    MACS3    : --nomodel --extsize 100 --keep-dup 1 -q 0.05')
            else:
                print(f'  [WARN] output files missing:')
                print(f'    {pp_np}')
                print(f'    {m3_np}')

    print(f'\n{"="*64}\n')


if __name__ == '__main__':
    main()
