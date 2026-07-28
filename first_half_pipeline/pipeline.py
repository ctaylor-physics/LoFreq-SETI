# pipeline.py
"""
Full pipeline first half: 
1. Build metadata CSVs from tarballs(temporarily saved unless otherwise specified)
2. Finds corresponding raw data files for each of the tarballs (one per station)
3. Run upchannlize on each station's data
4. Return 2 tuning files for each station


"""
import os
import argparse
import tempfile

from first_pipeline_scripts.get_metadata import build_csv
from first_pipeline_scripts.upchannelize import run as run_upchannelize

# ── Argument parsing ───────────────────────────────────────────────────────────
# These arguments are passed in from the command line when running the script,
# allowing the user to customize the processing without editing the code directly.
parser = argparse.ArgumentParser(description='Full LWA pipeline: metadata + upchannelizing')
parser.add_argument('--tar-path',  required=True)
parser.add_argument('--drx-path',  required=True)
parser.add_argument('--avg',       type=float, required=True)
parser.add_argument('--length',    type=int,   required=True)
parser.add_argument('--meta-dir',  default=None,
                    help='Where to write metadata CSVs (default: temp directory)')
parser.add_argument('--station', default=None,
                    help='Station name override (lwa1, lwasv, lwana) if mcs.host not in tarball')
args = parser.parse_args()

# Step 1: build metadata CSVs
meta_dir = args.meta_dir or tempfile.mkdtemp()
os.makedirs(meta_dir, exist_ok=True)
print(f"Writing metadata CSVs to: {meta_dir}")
build_csv(args.tar_path, output_dir=meta_dir, station=args.station)

# Step 2: run upchannelizing
run_upchannelize(
    tar_path=args.tar_path,
    drx_path=args.drx_path,
    avg=args.avg,
    length=args.length,
    meta_dir=meta_dir,
)


# run like this for all stations on the same date (e.g. DT006_260605_0600_0001):
# python3 pipeline.py \
#  --tar-path '/data/network/recent_data/lekness/DT006_260605_0600_0001/DT006_*.tgz' \ 
# --drx-path '/data/network/recent_data/lekness/DT006_260605_0600_0001/' \
# --avg 3.0 \
# --length 8388608


# python3 -m final_scripts.pipeline --tar-path '/data/network/recent_data/savin/alltar/DD002_8034.tgz' --drx-path '/data/network/recent_data/savin/drx_seti/059638_002859665' --avg 3.0 --length 8388608

# python3 -m final_scripts.pipeline --tar-path '/data/network/recent_data/savin/alltar/DD002_8051.tgz' --drx-path '/data/network/recent_data/savin/drx_seti/059659_003099308' --avg 3.0 --length 8388608 --station lwa1

# within legacy lsl venv: source .venv-legacy-lsl/bin/activate
# export LWA_STATION_OVERRIDE=lwa1
# python3 -m final_scripts.pipeline --tar-path '/data/network/recent_data/savin/alltar/DD002_8061.tgz' --drx-path '/data/network/recent_data/savin/drx_seti' --avg 3.0 --length 8388608 --station lwa1


# if you want to keep the metadata CSVs somewhere specific instead of a temp directory, run like this: 
# python3 pipeline.py \
# --tar-path '/data/network/recent_data/lekness/DT006_260605_0600_0001/DT006_*.tgz' \
# --drx-path '/data/network/recent_data/lekness/DT006_260605_0600_0001/' \
# --avg 3.0 \
# --length 8388608 \
# --meta-dir '/data/local/lekness/scripts_lillian/'



# additional changes: 
# add naming conventions for phase cal/flux cal/target to tuning files? 
    # or rename to be more like the original tarball name, e.g. DT006_260605_0600_0001_LWA1_tun1.h5
# take out the get_metadata step since we can get the metadata straight from the tarball
