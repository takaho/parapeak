import logging
import sys
from parapeak.cli import create_parser
from parapeak.peak_caller import run_peak_calling


def main():
    parser = create_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        stream=sys.stderr,
    )

    try:
        run_peak_calling(args)
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as exc:
        logging.getLogger(__name__).error(str(exc))
        sys.exit(1)


if __name__ == '__main__':
    main()
