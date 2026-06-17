import argparse

from parapeak import __version__


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='parapeak',
        description=(
            'NGS peak caller with parallel computation and GC-corrected statistics. '
            'Supports shallow (MiSeq) to deep coverage across ATAC-seq, '
            'genome-editing assays (CIRCLE-seq, GUIDE-seq), and similar experiments. '
            'Genome size is inferred from BAM headers; no reference FASTA required. '
            'BAM files do not need to be sorted or indexed.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '--version', action='version', version=f'parapeak {__version__}',
    )

    inp = parser.add_argument_group('Input')
    inp.add_argument(
        '-t', '--treated', nargs='+', required=True, metavar='BAM',
        help='Treated BAM/SAM files (sorting and indexing not required)',
    )
    inp.add_argument(
        '-c', '--control', nargs='+', default=None, metavar='BAM',
        help='Control/input BAM/SAM files',
    )
    inp.add_argument(
        '--blacklist', default=None, metavar='BED',
        help='Blacklist BED file (plain or gzip-compressed)',
    )

    out = parser.add_argument_group('Output')
    out.add_argument(
        '-o', '--output', default='parapeak_output', metavar='DIR',
        help='Output directory',
    )
    out.add_argument(
        '-n', '--name', default='parapeak', metavar='NAME',
        help='Output file prefix',
    )
    out.add_argument(
        '--signal-value',
        choices=['fold_enrichment', 'pvalue', 'pileup'],
        default='fold_enrichment',
        metavar='TYPE',
        help=(
            'Value written to the signal BED output for each significant peak. '
            '"fold_enrichment" (default): mean (treated + pc) / (scaled control + pc). '
            '"pvalue": max -log10 p-value from NB and Z-score tests at the peak. '
            '"pileup": max treated read count at the peak summit.'
        ),
    )
    out.add_argument(
        '--bigwig-bin', type=int, default=0, metavar='BP',
        help=(
            'Enable bigWig pileup output and set its bin size in bp (requires pyBigWig). '
            '0 (default) disables bigWig output. '
            'The effective bin size is rounded up to the nearest multiple of --bin-size. '
            'Values are treated pileup minus the genome-wide mean; '
            'only bins with a positive signal are written.'
        ),
    )

    param = parser.add_argument_group('Algorithm parameters')
    param.add_argument(
        '--local-window', type=int, default=1000, metavar='BP',
        help='Local background window size in bp',
    )
    param.add_argument(
        '--fragment-size', type=int, default=None, metavar='BP',
        help='Fragment size for read extension. '
             'Estimated from paired-end insert size if not given; falls back to 200.',
    )
    param.add_argument(
        '--bin-size', type=int, default=10, metavar='BP',
        help='Bin size for pileup in bp',
    )
    param.add_argument(
        '-q', '--qvalue', type=float, default=0.05, metavar='FLOAT',
        help='Q-value threshold applied to both NB and Z-score methods',
    )
    param.add_argument(
        '--min-length', type=int, default=200, metavar='BP',
        help='Minimum peak length in bp',
    )
    param.add_argument(
        '--max-gap', type=int, default=30, metavar='BP',
        help='Maximum gap between significant bins for merging',
    )

    qc = parser.add_argument_group('Read QC filters')
    qc.add_argument(
        '--min-mapq', type=int, default=20, metavar='INT',
        help='Minimum mapping quality (MAPQ). Reads below this threshold are discarded.',
    )
    qc.add_argument(
        '--min-fragment', type=int, default=10, metavar='BP',
        help='Minimum paired-end insert size accepted from TLEN. '
             'Smaller inserts are treated as single-end.',
    )
    qc.add_argument(
        '--max-fragment', type=int, default=2000, metavar='BP',
        help='Maximum paired-end insert size accepted from TLEN. '
             'Larger inserts are treated as single-end.',
    )
    qc.add_argument(
        '--keep-dup', type=int, default=1, metavar='INT',
        help='Maximum number of reads/fragments kept at the exact same '
             '(chrom, start, end) coordinate. Matches macs3\'s default '
             'behaviour of collapsing PCR duplicates that were never '
             'marked with the SAM duplicate flag. Set to 0 to disable.',
    )

    filter_group = parser.add_argument_group(
        'Signal filters',
        description=(
            'Post-statistical filters on absolute signal strength. '
            'These are especially important for low-coverage (MiSeq) data '
            'where statistical tests alone can pass spurious peaks caused '
            'by stochastic zero-count control bins.'
        ),
    )
    filter_group.add_argument(
        '--min-count', type=float, default=5.0, metavar='FLOAT',
        help=(
            'Minimum pileup value at the peak summit after read extension. '
            'Peaks whose summit bin contains fewer than this many overlapping '
            'extended reads are discarded. '
            'Increase for shallow libraries (e.g., 3 for MiSeq, 10 for deep data).'
        ),
    )
    filter_group.add_argument(
        '--min-fold', type=float, default=2.0, metavar='FLOAT',
        help=(
            'Minimum fold enrichment over the pseudocount-corrected background. '
            'Applied after --pseudocount is added to the denominator, '
            'so fold = (treated + pc) / (scaled_control + pc). '
            'Values < 1 disable this filter.'
        ),
    )
    filter_group.add_argument(
        '--pseudocount', type=float, default=1.0, metavar='FLOAT',
        help=(
            'Pseudocount added to the control background before NB modelling '
            'and fold-enrichment calculation. Prevents zero-control regions '
            'from producing infinite fold enrichment.'
        ),
    )

    parser.add_argument(
        '-p', '--threads', type=int, default=1, metavar='INT',
        help='Number of parallel worker processes',
    )

    return parser
