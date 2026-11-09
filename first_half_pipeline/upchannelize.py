"""
upchannelize.py

Batch-processes raw LWA DRX tarballs into upchannelized filterbank HDF5 files
for BLISS/SETI analysis.

For each tarball matching a glob pattern:
  1. Looks up its corresponding raw DRX data file and station (LWA1, LWA-SV,
     or LWA-NA) using per-station metadata CSVs.
  2. Runs hdfWaterfall to upchannelize/average the raw DRX data into a
     waterfall HDF5 file, at the given averaging time and FFT length.
  3. Converts that waterfall file into two per-tuning filterbank HDF5 files
     (_tun1.h5, _tun2.h5), preserving the full frequency grid and flagging
     configurable whole coarse channels at each edge, with header attributes (fch1, foff, tsamp,
     source coordinates, etc.) required by blimpy/turboSETI.
  4. Deletes the intermediate waterfall file.

Made for use in the pipeline

# High frequency resolution: --avg 3.0 --length 8388608

# High time resolution: --avg 0.2 --length 4096

# Custom: --avg 1.0 --length 65536
"""


import os
import glob
import csv
import h5py
import sys
import numpy as np
import argparse
from lsl import astro as ast


SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
HDFWATERFALL = os.path.join(SCRIPT_DIR, 'hdfWaterfall.py')

# ── CSV loading ────────────────────────────────────────────────────────────────
def load_csv(path):
    """
    Read a metadata CSV file and return three parallel lists:
      - metafiles: tarball filenames
      - beam_ids:  beam ID for each observation
      - drx_files: corresponding DRX data filenames
    These lists are used to look up which DRX file belongs to each tarball.
    """
    metafiles, beam_ids, drx_files = [], [], []
    with open(path, mode='r') as f:
        for line in csv.DictReader(f):
            # lstrip() removes any leading whitespace that may be in the CSV
            metafiles.append(line['tarball'].lstrip())
            beam_ids.append(line['beamid'])
            drx_files.append(line['datafile'].lstrip())
    return metafiles, beam_ids, drx_files

def coarse_layout(nchans, num_coarse=32, edge_coarse=3):
    """Return layout metadata; valid_channel_stop is an exclusive index.

    Edges refer to array order, independent of the sign of the frequency step.
    """
    if nchans < 2:
        raise ValueError("At least two fine channels are required")
    if num_coarse <= 0:
        raise ValueError("--num-coarse must be positive")
    if edge_coarse < 0:
        raise ValueError("--edge-coarse must be nonnegative")
    if 2 * edge_coarse >= num_coarse:
        raise ValueError("--edge-coarse must leave at least one searchable coarse channel")
    if nchans % num_coarse:
        raise ValueError(
            f"Fine-channel count {nchans} must be divisible by --num-coarse {num_coarse}"
        )
    width = nchans // num_coarse
    return {
        "coarse_layout_version": 1,
        "num_coarse": num_coarse,
        "edge_coarse": edge_coarse,
        "fine_channels_per_coarse": width,
        "valid_channel_start": edge_coarse * width,
        "valid_channel_stop": nchans - edge_coarse * width,
        "edge_mask_value": -1.0,
    }


def add_coarse_arguments(parser):
    parser.add_argument('--num-coarse', type=int, default=32,
                        help='Total coarse channels; must divide FFT length (default: 32)')
    parser.add_argument('--edge-coarse', type=int, default=3,
                        help='Whole coarse channels excluded at EACH edge (default: 3)')


# ── Tuning writer ──────────────────────────────────────────────────────────────
def write_tuning(hf, filename, outfile_suffix, idset, freq_tun, t0_mjd,
                 num_coarse=32, edge_coarse=3):
    """Write a full-grid filterbank, retaining interior samples unchanged.

    Excluded edge samples are -1.0 with a uint8 mask of 1. Interior masks
    are 0. These sentinels are not a BLISS mask: downstream searches must
    explicitly select the valid coarse-channel interval stored on data.
    """
    if idset.ndim != 2:
        raise ValueError("Tuning data must have shape (time, frequency)")
    nchans = idset.shape[1]
    layout = coarse_layout(nchans, num_coarse, edge_coarse)
    if np.shape(freq_tun) != (nchans,):
        raise ValueError("Frequency axis must match the actual tuning channel count")
    valid_start = layout['valid_channel_start']
    valid_stop = layout['valid_channel_stop']
    outfile = os.path.splitext(os.path.basename(filename))[0].replace('-waterfall', '') + outfile_suffix
    print(f"Writing: {outfile}")
    print(f"Frequency channels: {nchans} preserved; {num_coarse} coarse channels "
          f"of {layout['fine_channels_per_coarse']} fine channels each")
    print(f"Excluded per edge: {edge_coarse} coarse channels "
          f"({100 * edge_coarse / num_coarse:.3f}%); "
          f"searchable fine-channel interval [{valid_start}, {valid_stop})")

    with h5py.File(outfile, "w") as hf_out:
        hf_out.attrs['CLASS'] = 'FILTERBANK'
        hf_out.attrs['VERSION'] = '1.0'
        dshape = (idset.shape[0], 1, nchans)
        dset = hf_out.create_dataset('data', shape=dshape, dtype=np.float32)
        dset_mask = hf_out.create_dataset('mask', shape=dshape, dtype='uint8')
        for dataset in (dset, dset_mask):
            for axis, label in enumerate(('time', 'feed_id', 'frequency')):
                dataset.dims[axis].label = label
        dset_mask.attrs['description'] = '0 = valid; 1 = excluded coarse-channel edge'

        # Bound memory independently of observation duration and FFT length.
        # Read only valid input channels; do not materialize the full waterfall.
        for row in range(idset.shape[0]):
            if valid_start:
                dset[row, 0, :valid_start] = -1.0
                dset[row, 0, valid_stop:] = -1.0
                dset_mask[row, 0, :valid_start] = 1
                dset_mask[row, 0, valid_stop:] = 1
            for start in range(valid_start, valid_stop, 262144):
                stop = min(start + 262144, valid_stop)
                dset[row, 0, start:stop] = idset[row, start:stop]
            # HDF5's default zero fill leaves valid mask samples unflagged.

        dset.attrs.update(layout)
        dset.attrs['machine_id'] = 0
        dset.attrs['telescope_id'] = -1
        dset.attrs['src_raj'] = hf['Observation1'].attrs['RA']
        dset.attrs['src_dej'] = hf['Observation1'].attrs['Dec']
        dset.attrs['az_start'] = 0
        dset.attrs['za_start'] = 0
        dset.attrs['data_type'] = 1
        dset.attrs['fch1'] = freq_tun[0]
        dset.attrs['foff'] = freq_tun[1] - freq_tun[0]
        dset.attrs['nchans'] = nchans
        dset.attrs['nbeams'] = 1
        dset.attrs['ibeam'] = -1
        dset.attrs['nbits'] = 32
        dset.attrs['tstart'] = t0_mjd
        dset.attrs['tsamp'] = hf['Observation1'].attrs['tInt']
        dset.attrs['nifs'] = 1
        dset.attrs['source_name'] = hf['Observation1'].attrs['TargetName']
        dset.attrs['rawdatafile'] = os.path.basename(filename).split('-')[0]

    print(f"Finished writing {outfile}")


# ── Waterfall converter ────────────────────────────────────────────────────────
def convert_waterfall(waterfall_file, num_coarse=32, edge_coarse=3):
    """
    Open a waterfall HDF5 file produced by hdfWaterfall and write out one
    filterbank-format HDF5 file per tuning (two tunings per observation).

    Each output preserves all fine channels and flags whole coarse-channel edges.
    """
    print(f"\nConverting: {waterfall_file}")
    with h5py.File(waterfall_file, 'r') as hf:

        # Get the observation start time and convert it to MJD
        # time[0][0] is the integer part and time[0][1] is the fractional part
        time        = hf['Observation1/time']
        start_t     = time[0][0] + time[0][1]
        start_t_mjd = ast.jd_to_mjd(ast.unix_to_utcjd(start_t))
        print(f"Observation start time (MJD): {start_t_mjd}")

        # Read the Stokes I data for each tuning
        # Shape is (time, frequency) for each
        id_set1 = hf['Observation1/Tuning1/I']
        id_set2 = hf['Observation1/Tuning2/I']

        # Read the frequency arrays for each tuning and convert from Hz to MHz
        freq_1 = hf['Observation1/Tuning1/freq'][()] / 1e6
        freq_2 = hf['Observation1/Tuning2/freq'][()] / 1e6

        print(f"Tuning 1 frequency range: {freq_1[0]:.3f} - {freq_1[-1]:.3f} MHz")
        print(f"Tuning 2 frequency range: {freq_2[0]:.3f} - {freq_2[-1]:.3f} MHz")

        # Process each tuning separately into its own output file
        tunings = [
            (id_set1, freq_1, f"_tun1.h5"),
            (id_set2, freq_2, f"_tun2.h5"),
        ]
        # Validate both actual channel axes before creating either output file.
        for idset, freq_tun, _ in tunings:
            if idset.ndim != 2 or np.shape(freq_tun) != (idset.shape[1],):
                raise ValueError("Tuning data and frequency axes do not match")
            coarse_layout(idset.shape[1], num_coarse, edge_coarse)
        for idset, freq_tun, suffix in tunings:
            write_tuning(hf, waterfall_file, suffix, idset, freq_tun, start_t_mjd,
                         num_coarse=num_coarse, edge_coarse=edge_coarse)


# ── Tarball processor ──────────────────────────────────────────────────────────
def process_tarball(tarball, datafile_path, avg, length, work_dir, station_label,
                    num_coarse=32, edge_coarse=3):
    """
    Run the full pipeline for a single tarball:
      1. Run hdfWaterfall to upchannelize the raw DRX data into a waterfall HDF5
      2. Convert the waterfall HDF5 into per-tuning filterbank files
      3. Delete the intermediate waterfall HDF5 file
    """
    print(f"\n{'='*60}")
    print(f"Station:  {station_label}")
    print(f"Tarball:  {os.path.basename(tarball)}")
    print(f"Datafile: {datafile_path}")

    # Check that the DRX data file actually exists before trying to process it
    if not os.path.exists(datafile_path):
        print("ERROR: datafile does not exist, skipping")
        return

    # Step 1: Run hdfWaterfall to upchannelize and average the raw DRX data.
    # -a sets the averaging time, -l sets the FFT length,
    # -m points to the tarball (metadata), -k points to the DRX data file.
    print(f"\nStep 1: Running hdfWaterfall (avg={avg}, length={length})")
    os.system(f'{sys.executable} {HDFWATERFALL} -a {avg} -l {length} -m {tarball} -k {datafile_path}')

    # The waterfall file is written to the current working directory
    # with '-waterfall.hdf5' appended to the DRX filename

    # hdfWaterfall writes the file without the station label
    waterfall_file_raw = os.path.join(work_dir, os.path.basename(datafile_path) + '-waterfall.hdf5')

    if not os.path.exists(waterfall_file_raw):
        print("ERROR: Waterfall file was not produced, check the raw file/scripts")
        return


    waterfall_file = os.path.join(work_dir, os.path.basename(datafile_path) + f'-{station_label}-waterfall.hdf5')

    # Rename it to include the station label
    os.rename(waterfall_file_raw, waterfall_file)
    print(f"Renamed waterfall file to: {os.path.basename(waterfall_file)}")


    # Check that hdfWaterfall actually produced the output file
    
    # Step 2: Convert the waterfall HDF5 into per-tuning filterbank files,
    # preserving the full grid and flagging configured coarse-channel edges.
    print(f"\nStep 2: Converting waterfall to per-tuning filterbank files")
    convert_waterfall(waterfall_file, num_coarse=num_coarse, edge_coarse=edge_coarse)

    # Step 3: Remove the intermediate waterfall file to save disk space
    print(f"\nStep 3: Removing intermediate waterfall file")
    os.remove(waterfall_file)
    print("Done")


# ── Main ───────────────────────────────────────────────────────────────────────
def run(tar_path, drx_path, avg, length, meta_dir=".", num_coarse=32, edge_coarse=3):
    """
    Main function to run the full upchannelization and conversion pipeline.

    This function is called at the end of the script with the command line arguments.
    It loads the metadata CSVs, loops over the tarballs, determines their station,
    and processes each one through the pipeline.
    """
    coarse_layout(length, num_coarse, edge_coarse)
    metacsv1 = os.path.join(meta_dir, 'metadata_lwa1.csv')
    metacsv2 = os.path.join(meta_dir, 'metadata_lwa-sv.csv')
    metacsv3 = os.path.join(meta_dir, 'metadata_lwa-na.csv')
    # Load both metadata CSVs once before the loop so we only read them once
    metafiles1, beam_ids1, drx_files1 = load_csv(metacsv1)
    metafiles2, beam_ids2, drx_files2 = load_csv(metacsv2)
    metafiles3, beam_ids3, drx_files3 = load_csv(metacsv3)

    # The working directory is where hdfWaterfall will write the waterfall file
    work_dir = os.getcwd()
    tarballs = sorted(glob.glob(tar_path))
    print(f"\nFound {len(tarballs)} tarballs matching: {tar_path}")

# Loop over each job (in this script there is always one job, defined by the
# command line arguments, but the JOBS list structure makes it easy to extend)
    for tarball in tarballs:

        # Find all tarballs matching the glob pattern for this job
        tarball_base = os.path.basename(tarball)
        # avg      = job['avg']
        # length   = job['length']


        # Loop over each tarball and figure out which station it belongs to

        if tarball_base in metafiles1:
            # This tarball is from LWA1
            ind           = metafiles1.index(tarball_base)
            datafile_path = os.path.join(drx_path, drx_files1[ind])
            process_tarball(tarball, datafile_path, avg, length, work_dir, "LWA1",
                            num_coarse=num_coarse, edge_coarse=edge_coarse)

        elif tarball_base in metafiles2:
            # This tarball is from LWA-SV
            ind           = metafiles2.index(tarball_base)
            datafile_path = os.path.join(drx_path, drx_files2[ind])
            process_tarball(tarball, datafile_path, avg, length, work_dir, "LWA-SV",
                            num_coarse=num_coarse, edge_coarse=edge_coarse)
        elif tarball_base in metafiles3:
            # This tarball is from LWA-NA
            ind           = metafiles3.index(tarball_base)
            datafile_path = os.path.join(drx_path, drx_files3[ind])
            process_tarball(tarball, datafile_path, avg, length, work_dir, "LWA-NA",
                            num_coarse=num_coarse, edge_coarse=edge_coarse)

        else:
            # The tarball wasn't found in either station's metadata list
            print(f"\nWARNING: {tarball_base} not found in either station's metadata list, skipping")

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process LWA SETI observations')
    parser.add_argument('--tar-path', required=True,
                        help='Glob pattern for .tgz files')
    parser.add_argument('--drx-path', required=True,
                        help='Path to directory containing raw DRX data files')
    parser.add_argument('--avg',    type=float, required=True,
                        help='Averaging time in seconds')
    parser.add_argument('--length', type=int,   required=True,
                        help='FFT length')
    parser.add_argument('--meta-dir', default='.',  
                        help='Directory containing metadata CSVs (default: current directory)')
    add_coarse_arguments(parser)
    args = parser.parse_args()
    try:
        coarse_layout(args.length, args.num_coarse, args.edge_coarse)
    except ValueError as exc:
        parser.error(str(exc))

    run(
        tar_path=args.tar_path,
        drx_path=args.drx_path,
        avg=args.avg,
        length=args.length,
        meta_dir=args.meta_dir,
        num_coarse=args.num_coarse,
        edge_coarse=args.edge_coarse,
    )
