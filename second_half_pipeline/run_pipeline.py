"""
run_pipeline.py

End-to-end LWA coincidence-search pipeline.

Given up to 6 .h5 files (2 or 3 stations, 2 tunings each: tun1 and tun2),
runs, per input file:

    1. lwa_bliss_bp_gen.py   -> corrected HDF5 data and saved bandpass
    2. run_bliss.py          -> direct .dat search and correctly labeled hits CSV

then, per tuning (tun1 / tun2) separately:

    3. anti_coincidence.py       -> matched-hit CSVs across the 2 or 3 stations
    4. plot_pair_coincidence.py  (2 stations)
       or plot_triple_coincidence.py (3 stations)
                                  -> waterfall stamp PNGs/PDF

Station and tuning are inferred from each filename:
    station: "LWA-SV" -> SV, "LWA-NA" -> NA, "LWA1"/"LWA-1" -> ONE
    tuning:  "tun1" -> 1, "tun2" -> 2

This script does not reimplement any of the science logic in the underlying
scripts -- it only calls them (via subprocess, using their existing CLIs) in
the right order with the right arguments, and stitches the file names
together in between.

Usage
-----
python3 -m second_half_pipeline.run_pipeline FILE1.h5 FILE2.h5 FILE3.h5 FILE4.h5 \\
    [--freq_tol 10] [--drift_tol 0.8] [--width 524] \\
    [--bliss-executable /path/to/bliss_find_hits] [--fit-clip] \\
    [--outdir pipeline_results] [--scripts_dir .] [--force]


"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
import h5py

if __package__:
    from .run_bliss import add_search_arguments
else:
    from run_bliss import add_search_arguments


STATION_ORDER = ['ONE', 'NA', 'SV']   # fixed priority order used by anti_coincidence.py
STATION_LABELS = {'ONE': 'LWA-1', 'NA': 'LWA-NA', 'SV': 'LWA-SV'}
STATION_FLAGS = {'ONE': '--one', 'NA': '--na', 'SV': '--sv'}

STATION_PATTERNS = [
    (re.compile(r'lwa-sv', re.IGNORECASE), 'SV'),
    (re.compile(r'lwa-na', re.IGNORECASE), 'NA'),
    (re.compile(r'lwa-?1(?!\d)', re.IGNORECASE), 'ONE'),
]
TUNING_PATTERN = re.compile(r'tun(1|2)', re.IGNORECASE)


# ── filename parsing ────────────────────────────────────────────────────────

def parse_station(filename):
    for pattern, code in STATION_PATTERNS:
        if pattern.search(filename):
            return code
    return None


def parse_tuning(filename):
    m = TUNING_PATTERN.search(filename)
    if m:
        return int(m.group(1))
    return None


def group_inputs(h5_files):
    """
    Returns: { station_code: { tuning_int: abs_h5_path } }
    Raises a clear error if any file can't be parsed, or if the resulting
    set of stations/tunings isn't a valid 2- or 3-station, 2-tuning-each set.
    """
    grouped = defaultdict(dict)
    problems = []

    for f in h5_files:
        base = os.path.basename(f)
        station = parse_station(base)
        tuning = parse_tuning(base)
        if station is None or tuning is None:
            problems.append(
                f"  {f}  (station parsed: {station}, tuning parsed: {tuning})"
            )
            continue
        if tuning in grouped[station]:
            problems.append(
                f"  {f}  -> duplicate: station {station} tuning {tuning} already "
                f"assigned to {grouped[station][tuning]}"
            )
            continue
        grouped[station][tuning] = os.path.abspath(f)

    if problems:
        raise ValueError(
            "Could not confidently assign station/tuning for the following input file(s):\n"
            + "\n".join(problems)
            + "\n\nExpected filenames to contain one of LWA-SV / LWA-NA / LWA1 (or LWA-1) "
              "and one of tun1 / tun2."
        )

    n_stations = len(grouped)
    if n_stations not in (2, 3):
        raise ValueError(
            f"Expected files from 2 or 3 stations, but found {n_stations}: "
            f"{list(grouped.keys())}"
        )

    for station, tunings in grouped.items():
        missing = {1, 2} - set(tunings.keys())
        if missing:
            raise ValueError(
                f"Station {station} is missing tuning(s) {sorted(missing)}. "
                f"Every station needs both tun1 and tun2."
            )

    return dict(grouped)


# ── subprocess helper ───────────────────────────────────────────────────────

def run(cmd, cwd=None, step_label=""):
    print(f"\n{'='*70}\n[RUN] {step_label}\n{'  '.join(cmd)}\n{'='*70}")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed ({step_label}): {' '.join(cmd)}")


# ── pipeline stages ─────────────────────────────────────────────────────────

def make_bandpass(h5_path, workdir, scripts_dir, force, fit_clip=False,
                  fit_clip_sigma=10.0, fit_clip_window=101, fit_clip_max_width=8):
    basename = Path(h5_path).stem
    bp_path = os.path.join(workdir, f"{basename}_bpmodel.f32")
    os.makedirs(workdir, exist_ok=True)
    # Reuse only completed HDF5 corrections, never a possibly stale .f32 file.
    with h5py.File(h5_path, 'r') as f:
        complete = ('_bandpass_work' not in f and 'uncorrected' in f and
                    'bandpass_model' in f and 'data' in f and
                    f['data'].attrs.get('bandpass_correction_version') == 1)
        if complete and not force:
            attrs = f['bandpass_model'].attrs
            desired = {'fit_clip_enabled': fit_clip, 'fit_clip_sigma': fit_clip_sigma,
                       'fit_clip_window': fit_clip_window, 'fit_clip_max_width': fit_clip_max_width}
            if bool(attrs.get('fit_clip_enabled', False)) != fit_clip or (
                fit_clip and any(attrs.get(k) != v for k, v in desired.items())
            ):
                raise ValueError('Requested fit clipping differs from the saved fit; use --force to refit originals')
            print(f'[skip] completed HDF5 bandpass correction: {h5_path}')
            f['bandpass_model'][:].tofile(bp_path)
            return bp_path
    script = os.path.join(scripts_dir, 'lwa_bliss_bp_gen.py')
    command = [sys.executable, script, h5_path, '--output', bp_path,
               '--fit-clip-sigma', str(fit_clip_sigma), '--fit-clip-window', str(fit_clip_window),
               '--fit-clip-max-width', str(fit_clip_max_width)]
    if fit_clip:
        command.append('--fit-clip')
    if force:
        command.append('--force')
    run(command, cwd=workdir, step_label=f'bandpass correction for {basename}')
    if not os.path.exists(bp_path):
        raise RuntimeError(f'Bandpass profile was not created: {bp_path}')
    return bp_path


def make_hits(h5_path, workdir, scripts_dir, options):
    script = os.path.join(scripts_dir, 'run_bliss.py')
    command = [sys.executable, script, '--h5', h5_path, '--outdir', workdir,
               '--bliss-executable', options.bliss_executable, '--snr', str(options.snr),
               '--min-drift', str(options.min_drift), '--max-drift', str(options.max_drift),
               '--drift-step', str(options.drift_step), '--distance', str(options.distance)]
    if options.device is not None:
        command += ['--device', options.device]
    if options.force:
        command.append('--force')
    run(command, step_label=f'direct BLISS search for {Path(h5_path).stem}')
    hits_path = os.path.join(workdir, f'{Path(h5_path).stem}_bliss_hits.csv')
    if not os.path.exists(hits_path):
        raise RuntimeError(f'BLISS runner did not produce {hits_path}')
    return hits_path


def run_anti_coincidence(hits_by_station, tuning_outdir, scripts_dir,
                          freq_tol, drift_tol, force):
    present = [s for s in STATION_ORDER if s in hits_by_station]

    script = os.path.join(scripts_dir, 'anti_coincidence.py')
    cmd = [sys.executable, script]
    for s in present:
        cmd += [STATION_FLAGS[s], hits_by_station[s]]
    cmd += [
        '--freq_tol', str(freq_tol),
        '--drift_tol', str(drift_tol),
        '--outdir', tuning_outdir,
    ]

    if len(present) == 2:
        name1, name2 = present
        expected = {
            name1: os.path.join(tuning_outdir, f'hits_{name1}_{name2}_{name1}.csv'),
            name2: os.path.join(tuning_outdir, f'hits_{name1}_{name2}_{name2}.csv'),
        }
    else:
        expected = {
            'ONE': os.path.join(tuning_outdir, 'hits_all_three_ONE.csv'),
            'NA':  os.path.join(tuning_outdir, 'hits_all_three_NA.csv'),
            'SV':  os.path.join(tuning_outdir, 'hits_all_three_SV.csv'),
        }

    os.makedirs(tuning_outdir, exist_ok=True)
    run(cmd, step_label=f"anti_coincidence ({'/'.join(present)})")

    missing = {s: p for s, p in expected.items() if not os.path.exists(p)}
    if missing:
        raise RuntimeError(f"anti_coincidence.py did not produce expected output(s): {missing}")

    return expected


def run_plotting(present, matched_csvs, h5_by_station, tuning_outdir, scripts_dir,
                  width, freq_tol, drift_tol, force):
    plots_dir = os.path.join(tuning_outdir, 'plots')
    # Matching and plots are regenerated so old schemas/settings cannot be reused.
    for pattern in ('pair_hit_*.png', 'triple_hit_*.png', 'pair_coincidence_stamps.pdf',
                    'triple_coincidence_stamps.pdf'):
        for path in Path(plots_dir).glob(pattern):
            path.unlink()

    if len(present) == 2:
        name1, name2 = present
        pdf_path = os.path.join(plots_dir, 'pair_coincidence_stamps.pdf')
        script = os.path.join(scripts_dir, 'plot_pair_coincidence.py')
        cmd = [
            sys.executable, script,
            '--csv_a', matched_csvs[name1], '--h5_a', h5_by_station[name1],
            '--label_a', STATION_LABELS[name1],
            '--csv_b', matched_csvs[name2], '--h5_b', h5_by_station[name2],
            '--label_b', STATION_LABELS[name2],
            '--width', str(width),
            '--freq_tol', str(freq_tol),
            '--drift_tol', str(drift_tol),
            '--outdir', plots_dir,
        ]
        run(cmd, step_label=f"plot_pair_coincidence ({name1}/{name2})")
    else:
        pdf_path = os.path.join(plots_dir, 'triple_coincidence_stamps.pdf')
        script = os.path.join(scripts_dir, 'plot_triple_coincidence.py')
        cmd = [
            sys.executable, script,
            '--na_csv',  matched_csvs['NA'],  '--na_h5',  h5_by_station['NA'],
            '--sv_csv',  matched_csvs['SV'],  '--sv_h5',  h5_by_station['SV'],
            '--one_csv', matched_csvs['ONE'], '--one_h5', h5_by_station['ONE'],
            '--width', str(width),
            '--freq_tol', str(freq_tol),
            '--drift_tol', str(drift_tol),
            '--outdir', plots_dir,
        ]
        run(cmd, step_label="plot_triple_coincidence (NA/SV/ONE)")

    return plots_dir


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full LWA multi-station coincidence pipeline.")
    parser.add_argument('h5_files', nargs='+', help='4 or 6 input .h5 files (2 or 3 stations x tun1/tun2)')
    parser.add_argument('--freq_tol', type=float, default=10.0, help='Frequency tolerance in Hz (default: 10)')
    parser.add_argument('--drift_tol', type=float, default=0.8, help='Drift rate tolerance in Hz/s (default: 0.8)')
    parser.add_argument('--width', type=int, default=524, help='Postage-stamp channel width for plots (default: 524)')
    add_search_arguments(parser)
    parser.add_argument('--fit-clip', action='store_true', help='Enable optional fitting-spectrum clipping')
    parser.add_argument('--fit-clip-sigma', type=float, default=10.0)
    parser.add_argument('--fit-clip-window', type=int, default=101)
    parser.add_argument('--fit-clip-max-width', type=int, default=8)
    parser.add_argument('--outdir', default='pipeline_results', help='Top-level output directory')
    parser.add_argument('--scripts_dir', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Directory containing the other pipeline scripts (default: this script\'s directory)')
    parser.add_argument('--force', action='store_true',
                        help='Re-run every stage even if expected output files already exist')
    args = parser.parse_args()

    if len(args.h5_files) not in (4, 6):
        parser.error(f"Expected 4 or 6 input .h5 files (2 or 3 stations x 2 tunings), got {len(args.h5_files)}.")

    executable = shutil.which(args.bliss_executable)
    if executable is None:
        parser.error('BLISS executable not found; set --bliss-executable')
    args.bliss_executable = os.path.abspath(executable)
    for path in args.h5_files:
        if not os.path.isfile(path):
            parser.error(f'Input does not exist: {path}')

    outdir = os.path.abspath(args.outdir)
    scripts_dir = os.path.abspath(args.scripts_dir)

    print("Parsing station/tuning from filenames...")
    grouped = group_inputs(args.h5_files)
    present_stations = [s for s in STATION_ORDER if s in grouped]
    mode = 'pair' if len(present_stations) == 2 else 'triple'
    print(f"Stations found: {present_stations}  ->  mode: {mode}")
    for s in present_stations:
        print(f"  {s}: tun1={grouped[s][1]}")
        print(f"  {s}: tun2={grouped[s][2]}")

    for tuning in (1, 2):
        print(f"\n{'#'*70}\n# TUNING {tuning}\n{'#'*70}")
        tuning_outdir = os.path.join(outdir, f'tun{tuning}')
        hits_workdir = os.path.join(tuning_outdir, 'hits')
        os.makedirs(hits_workdir, exist_ok=True)

        hits_by_station = {}
        h5_by_station = {}
        for station in present_stations:
            h5_path = grouped[station][tuning]
            h5_by_station[station] = h5_path

            make_bandpass(h5_path, hits_workdir, scripts_dir, args.force, args.fit_clip,
                          args.fit_clip_sigma, args.fit_clip_window, args.fit_clip_max_width)
            hits_path = make_hits(h5_path, hits_workdir, scripts_dir, args)
            hits_by_station[station] = hits_path

        matched_csvs = run_anti_coincidence(hits_by_station, tuning_outdir, scripts_dir,
                                            args.freq_tol, args.drift_tol, args.force)

        plots_dir = run_plotting(present_stations, matched_csvs, h5_by_station, tuning_outdir,
                                 scripts_dir, args.width, args.freq_tol, args.drift_tol, args.force)

        print(f"\nTuning {tuning} done. Plots in: {plots_dir}")

    print(f"\nPipeline complete. All results under: {outdir}")


if __name__ == "__main__":
    main()

# Example usage:
# python3 run_pipeline.py
#     frb_set_two/059613_002172869-LWA-SV_tun1.h5 frb_set_two/059613_002172869-LWA-SV_tun2.h5
#     frb_set_two/059613_002567249-LWA1_tun1.h5   frb_set_two/059613_002567249-LWA1_tun2.h5
#     --freq_tol 10 --drift_tol 0.8 --width 524 --outdir results_frb_set_three


# python3 -m second_half_pipeline.run_pipeline ../frb_data/frb_8/059639_002868043-LWA1_tun1.h5 ../frb_data/frb_8/059639_002868043-LWA1_tun2.h5 ../frb_data/frb_8/059639_002431255-LWA-SV_tun1.h5 ../frb_data/frb_8/059639_002431255-LWA-SV_tun2.h5 --freq_tol 30 --drift_tol 0.8 --width 524 --outdir results_frb_7
