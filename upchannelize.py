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
     (_tun1.h5, _tun2.h5), trimming the outer 10% of frequency channels
     from each edge and writing the header attributes (fch1, foff, tsamp,
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

# ── Tuning writer ──────────────────────────────────────────────────────────────
def write_tuning(hf, filename, outfile_suffix, idset, freq_tun, t0_mjd):
    """
    Write a single tuning to its own filterbank-format HDF5 file.

    Before writing, trims the outer 10% of frequency channels from each edge,
    keeping only the inner 80% where the data quality is best.

    Inputs:
      hf            - the open waterfall HDF5 file (used to read observation metadata)
      filename      - path to the waterfall file (used to build the output filename)
      outfile_suffix - string appended to the output filename e.g. '_tun1.h5'
      idset         - 2D array of Stokes I data (time x frequency) - get from the waterfall conversion
      freq_tun      - 1D array of frequency values in MHz for this tuning
      t0_mjd        - observation start time in MJD
    """

    # Build the output filename from the input filename + suffix
    outfile = os.path.splitext(os.path.basename(filename))[0].replace('-waterfall', '') + outfile_suffix
    print(f"Writing: {outfile}")

    # ── Trim to inner 80% of frequency channels ────────────────────────────────
    # The edges of the band often have lower quality data due to the filter rolloff,
    # so we cut 10% from each side, keeping the central 80%.
    edge_size = int(round(idset.shape[1] * 0.1))
    idset_cut = idset[:, edge_size:-edge_size]   # trim time x frequency array
    freq_cut  = freq_tun[edge_size:-edge_size]   # trim frequency array to match

    print(f"Frequency channels: {idset.shape[1]} total → {idset_cut.shape[1]} kept "
          f"(trimmed {edge_size} channels from each edge)")

    # ── Create the output HDF5 file ────────────────────────────────────────────
    hf_out = h5py.File(outfile, "w")

    # Top-level attributes identify this as a filterbank file
    hf_out.attrs['CLASS']   = 'FILTERBANK'
    hf_out.attrs['VERSION'] = '1.0'

    # The data shape is (time, feed, frequency) -- feed is always 1 for LWA
    # We use the trimmed shape here so the dataset matches the cut data
    dshape = (idset_cut.shape[0], 1, idset_cut.shape[1])
    print(f"Output dataset shape (time, feed, frequency): {dshape}")

    # Create the main data dataset and a corresponding mask dataset
    dset      = hf_out.create_dataset("data", shape=dshape, dtype=idset.dtype)
    dset_mask = hf_out.create_dataset("mask", shape=dshape, dtype="uint8")

    # Label each axis so downstream tools know which dimension is which
    dset.dims[2].label = b"frequency"
    dset.dims[1].label = b"feed_id"
    dset.dims[0].label = b"time"

    dset_mask.dims[2].label = b"frequency"
    dset_mask.dims[1].label = b"feed_id"
    dset_mask.dims[0].label = b"time"

    # ── Write data ─────────────────────────────────────────────────────────────
    # Write the trimmed Stokes I data into the dataset
    dset[:,0,:]      = idset_cut
    # Mask is all zeros (no channels flagged) -- shape must match the trimmed data
    dset_mask[:,0,:] = np.zeros(idset_cut.shape, dtype='uint8')

    # ── Write header attributes ────────────────────────────────────────────────
    # These attributes are required by blimpy/turboSETI to interpret the data correctly.
    # Several of them are affected by the frequency trim and must use the cut values.
    dset.attrs['machine_id']   = 0       # 0 = unknown machine
    dset.attrs['telescope_id'] = -1      # -1 = unknown telescope
    dset.attrs['src_raj']      = hf['Observation1'].attrs['RA']    # right ascension
    dset.attrs['src_dej']      = hf['Observation1'].attrs['Dec']   # declination
    dset.attrs['az_start']     = 0       # azimuth at start (unknown)
    dset.attrs['za_start']     = 0       # zenith angle at start (unknown)
    dset.attrs['data_type']    = 1       # 1 = filterbank data

    # fch1 is the frequency of the first channel -- must use freq_cut[0] not freq_tun[0]
    # since we trimmed the edges
    dset.attrs['fch1']         = freq_cut[0]

    # foff is the channel spacing in MHz -- unchanged by the trim since we kept
    # evenly spaced channels, just fewer of them
    dset.attrs['foff']         = freq_cut[1] - freq_cut[0]

    # nchans must reflect the trimmed number of channels, not the original
    dset.attrs['nchans']       = idset_cut.shape[1]

    dset.attrs['nbeams']       = 1       # number of beams
    dset.attrs['ibeam']        = -1      # beam index (-1 = unknown)
    dset.attrs['nbits']        = 32      # 32-bit float data
    dset.attrs['tstart']       = t0_mjd  # observation start time in MJD
    dset.attrs['tsamp']        = hf['Observation1'].attrs['tInt']         # time per sample
    dset.attrs['nifs']         = 1       # number of IFs (polarizations stored)
    dset.attrs['source_name']  = hf['Observation1'].attrs['TargetName']   # target name
    dset.attrs['rawdatafile']  = os.path.basename(filename).split('-')[0] # original filename

    hf_out.close()
    print(f"Finished writing {outfile}")


# ── Waterfall converter ────────────────────────────────────────────────────────
def convert_waterfall(waterfall_file):
    """
    Open a waterfall HDF5 file produced by hdfWaterfall and write out one
    filterbank-format HDF5 file per tuning (two tunings per observation).

    Each output file contains only the inner 80% of frequency channels.
    """
    print(f"\nConverting: {waterfall_file}")
    hf = h5py.File(waterfall_file, 'r')

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
    for idset, freq_tun, suffix in tunings:
        write_tuning(hf, waterfall_file, suffix, idset, freq_tun, start_t_mjd)

    hf.close()


# ── Tarball processor ──────────────────────────────────────────────────────────
def process_tarball(tarball, datafile_path, avg, length, work_dir, station_label):
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
    # trimming the outer 10% of frequency channels from each edge.
    print(f"\nStep 2: Converting waterfall to per-tuning filterbank files")
    convert_waterfall(waterfall_file)

    # Step 3: Remove the intermediate waterfall file to save disk space
    print(f"\nStep 3: Removing intermediate waterfall file")
    os.remove(waterfall_file)
    print("Done")


# ── Main ───────────────────────────────────────────────────────────────────────
def run(tar_path, drx_path, avg, length, meta_dir="."):
    """
    Main function to run the full upchannelization and conversion pipeline.

    This function is called at the end of the script with the command line arguments.
    It loads the metadata CSVs, loops over the tarballs, determines their station,
    and processes each one through the pipeline.
    """
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
            process_tarball(tarball, datafile_path, avg, length, work_dir, "LWA1")

        elif tarball_base in metafiles2:
            # This tarball is from LWA-SV
            ind           = metafiles2.index(tarball_base)
            datafile_path = os.path.join(drx_path, drx_files2[ind])
            process_tarball(tarball, datafile_path, avg, length, work_dir, "LWA-SV")
        elif tarball_base in metafiles3:
            # This tarball is from LWA-NA
            ind           = metafiles3.index(tarball_base)
            datafile_path = os.path.join(drx_path, drx_files3[ind])
            process_tarball(tarball, datafile_path, avg, length, work_dir, "LWA-NA")

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
    args = parser.parse_args()

    run(
        tar_path=args.tar_path,
        drx_path=args.drx_path,
        avg=args.avg,
        length=args.length,
        meta_dir=args.meta_dir,
    )
