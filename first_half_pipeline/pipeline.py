"""Build station metadata and convert raw DRX data to full-grid tuning files."""
import argparse
import os
import tempfile

if __package__:
    from .upchannelize import add_coarse_arguments, coarse_layout, run as run_upchannelize
else:
    from upchannelize import add_coarse_arguments, coarse_layout, run as run_upchannelize


def main():
    parser = argparse.ArgumentParser(description='Full LWA pipeline: metadata + upchannelizing')
    parser.add_argument('--tar-path', required=True)
    parser.add_argument('--drx-path', required=True)
    parser.add_argument('--avg', type=float, required=True)
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--meta-dir', default=None,
                        help='Where to write metadata CSVs (default: temp directory)')
    parser.add_argument('--station', default=None,
                        help='Station override (lwa1, lwasv, lwana) if mcs.host is absent')
    add_coarse_arguments(parser)
    args = parser.parse_args()
    try:
        coarse_layout(args.length, args.num_coarse, args.edge_coarse)
    except ValueError as exc:
        parser.error(str(exc))

    if __package__:
        from .get_metadata import build_csv
    else:
        from get_metadata import build_csv

    meta_dir = args.meta_dir or tempfile.mkdtemp()
    os.makedirs(meta_dir, exist_ok=True)
    print(f"Writing metadata CSVs to: {meta_dir}")
    build_csv(args.tar_path, output_dir=meta_dir, station=args.station)
    run_upchannelize(
        tar_path=args.tar_path,
        drx_path=args.drx_path,
        avg=args.avg,
        length=args.length,
        meta_dir=meta_dir,
        num_coarse=args.num_coarse,
        edge_coarse=args.edge_coarse,
    )


if __name__ == '__main__':
    main()
