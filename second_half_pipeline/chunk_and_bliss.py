'''
chunk_and_bliss.py
'''

import h5py
import numpy as np
import pandas as pd
import time
import os
import glob
import argparse

BLISS_DIR = '/home/ctaylor/bliss/build/bliss' # Note: this directory is on LWAUCF3. Change this variable to a different location if bliss is installed elsewhere

def make_overlapping_channel_chunks(nchan, chunk_size, min_overlap=2048):
    """
    Return [(start, stop), ...] channel slices with at least min_overlap
    channels shared between adjacent chunks.

    stop is exclusive, so use data[..., start:stop].
    """
    if chunk_size <= min_overlap:
        raise ValueError("chunk_size must be larger than min_overlap")

    chunks = []
    step = chunk_size - min_overlap

    start = 0
    while start < nchan:
        stop = min(start + chunk_size, nchan) # at the end the last chunk might hit the end of the channels
        chunks.append((start, stop))

        if stop == nchan:
            break

        start += step

    return chunks


def run_bliss_on_chunk(datafile, bpfile, nchans, outfile):
    """
    Handler to run bliss and convert the hits into dat files
    """
    ### Process Files
    outfile2 = f'{outfile}.bfh'

    ## Main Call
    bliss_cmd = f'{BLISS_DIR}/bliss_find_hits {datafile} -e {bpfile} --number-coarse 1 --nchan-per-coarse {nchans} --distance 7 -s 10 -md -3 -MD 3 -rs 1 -o {outfile2}'
    os.system(bliss_cmd) # gives mask that is the calculated bandpass profile (bpfile) for bliss to integrate

    ## Convert Output to .dat
    write_dat_cmd = f'{BLISS_DIR}/bliss_hits_to_dat -i {outfile2} -o {outfile2}.dat'
    os.system(write_dat_cmd)

    ### Clean Up
    os.remove(datafile) # chunk that is slice of the original file
    os.remove(bpfile) # bandpass profile for that chunk
    os.remove(outfile2) # intermediate bliss output file
    return

def chunk_h5py_and_bpm(filename, bpfile, chunks, prefix):
    """
    Divides the input filename into smaller pieces according to chunks,
        searches each chunk for technosignatures, then writes the interesting signal hits to a csv file

    Inputs:
        a .h5 filename, corresponding bandpass profile, and chunk locations.
        prefix: a string unique to this input file, used to name intermediate
                chunk/bliss output so that runs on different input files
                (e.g. in a multi-file pipeline) never collide in the same
                working directory.
    Returns:
        Writes hits.dat files.
    """

    # Monitoring
    time_0 = time.time()
    # Number of subfiles to build
    chunk_num = np.arange(len(chunks))
    ## load up bandpass
    bpm = np.fromfile(bpfile, dtype=np.float32)

    for i in chunk_num:
        ### Get start and stop indices
        start, stop = chunks[i]
        diff = stop-start
        outfile = f"{os.path.basename(filename).split('.')[0]}_{i+1}.h5"

        with h5py.File(filename, 'r') as f_in, h5py.File(outfile, 'w') as f_out:
            print(f"Writing Chunk {i}: {(time.time() - time_0):.2f}") # made time not inverted

            ### header info
            f_out.attrs['CLASS'] = 'FILTERBANK'
            f_out.attrs['VERSION'] = '1.0'

            ### Slice chunk data
            in_data = f_in['data'][:,:,start:stop]
            in_mask = f_in['mask'][:,:,start:stop] # this isn't used for anything rn

            ### Create new datasets
            ## data
            out_shape = in_data.shape
            out_data = f_out.create_dataset('data', shape=out_shape, dtype=np.float32)
            out_data.dims[0].label = b"time"
            out_data.dims[1].label = b"feed_id"
            out_data.dims[2].label = b"frequency"
            ## mask

            out_shape = in_mask.shape # should this be added?

            out_mask = f_out.create_dataset('mask', shape=out_shape, dtype=np.float32)
            out_mask.dims[0].label = b"time"
            out_mask.dims[1].label = b"feed_id"
            out_mask.dims[2].label = b"frequency"
            ## assign values
            out_data[:] = in_data
            out_mask[:] = np.zeros(out_shape, dtype=np.uint8)
            ## attributes
            f_in_attr = dict(f_in['data'].attrs.items())
            new_fch1 = f_in_attr['fch1'] + f_in_attr['foff'] * start
            f_in_attr['fch1'] = new_fch1
            f_in_attr['nchans'] = stop-start
            out_data.attrs.update(f_in_attr)

            ## write out trimmed bandpass
            out_bpfile = f"{os.path.basename(bpfile).split('.')[0]}_{i+1}.f32"
            bpm[start:stop].tofile(out_bpfile)

        ## run bliss
        print(f"Running Bliss on Chunk {i}")
        # NOTE: output name now includes `prefix` (derived from the input
        # filename) instead of the original hardcoded "lwa_bliss_test" so
        # that multiple files processed in the same working directory don't
        # overwrite each other's intermediate/output files.
        run_bliss_on_chunk(outfile, out_bpfile, diff, f"{prefix}_chunk_{i+1:03}")


    return print(f"Done {(time.time() - time_0):.2f} Elapsed") # time not inverted anymore


def combine_datfiles(filenames, outfile):
    """
    Take all chunk .dat files created by bliss and combine them into a single file
    """
    # .dat cols
    cols = ["Top_Hit_#", "Drift_Rate",  "SNR", "Uncorrected_Frequency", "Corrected_Frequency", "Index", "freq_start", "freq_end", "SEFD_freq", "Coarse_Channel_Number", "Full_number_of_hits"]
    # write concat file
    full_table = pd.concat([pd.read_csv(file, header=8, sep='\t', names=cols) for file in filenames], ignore_index=True)
    full_table.to_csv(outfile)
    # delete split files
    for f in filenames:
        try:
           os.remove(f)
        except OSError as e:
            print(f"Error deleting {f}: {e}")

    return


def process_file(h5_path, bp_path, outdir, nchan, chunksize, min_overlap):
    """
    Run the full chunk -> bliss -> combine sequence for one input .h5 file,
    writing the combined hits CSV (and, transiently, all intermediate
    per-chunk files) into `outdir`.

    Returns the path to the final combined hits CSV.
    """
    os.makedirs(outdir, exist_ok=True)
    orig_cwd = os.getcwd()
    os.chdir(outdir)
    try:
        basename = os.path.basename(h5_path).split('.')[0]
        outfile = f"{basename}_bliss_hits.csv"

        chunks = make_overlapping_channel_chunks(nchan, chunksize, min_overlap)

        print(f'Starting File Chunking for {basename}')
        chunk_h5py_and_bpm(h5_path, bp_path, chunks, prefix=basename)

        print('Combining output')
        dat_filenames = sorted(glob.glob(f'{basename}_chunk_*.dat'))
        combine_datfiles(dat_filenames, outfile)

        return os.path.join(outdir, outfile)
    finally:
        os.chdir(orig_cwd)


def main():
    parser = argparse.ArgumentParser(
        description='Chunk an .h5 waterfall file, run bliss hit-finding on each chunk, '
                    'and combine the results into one hits CSV.'
    )
    parser.add_argument('--h5', required=True, help='Input .h5 waterfall file')
    parser.add_argument('--bp', required=True, help='Bandpass profile (.f32) file matching --h5')
    parser.add_argument('--outdir', default='.',
                        help='Directory to write the combined hits CSV (and intermediates) to (default: current dir)')
    parser.add_argument('--nchan', type=int, default=6710886,
                        help='Total number of frequency channels in the input file (default: 6710886)')
    parser.add_argument('--chunksize', type=int, default=2**18,
                        help='Channel width of each chunk (default: 2^18)')
    parser.add_argument('--min_overlap', type=int, default=2048,
                        help='Minimum channel overlap between adjacent chunks (default: 2048)')
    args = parser.parse_args()

    h5_path = os.path.abspath(args.h5)
    bp_path = os.path.abspath(args.bp)
    outdir  = os.path.abspath(args.outdir)

    outfile_path = process_file(h5_path, bp_path, outdir, args.nchan, args.chunksize, args.min_overlap)
    print(f"\nHits CSV written to: {outfile_path}")


if __name__ == "__main__":
    main()

# Example usage:
# python chunk_and_bliss.py --h5 059659_002631675-LWA-SV_tun2.h5 --bp 059659_002631675-LWA-SV_tun2_bpmodel.f32 --outdir ./tun2_hits_test
