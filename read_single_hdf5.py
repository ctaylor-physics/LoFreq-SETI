"""
This script can be used for inspection of single files

Stand-alone, hardcoded script to read in a single LWA HDF5 file and write out the two tunings to separate filterbank HDF5 files.
This script was a temporary solution until the LWA HDF5 files are updated to include the necessary metadata for the filterbank 
format. The script extracts the I values from each tuning and writes them to new HDF5 files with the appropriate attributes.

Ancestor of upchannelize.py's convert_waterfall() and write_tuning() functions, which are now used in the pipeline to process multiple files and tunings.
Note: this script does not trim the edges of the data, but the pipeline does. 
"""

import os
import h5py
import numpy as np
from lsl import astro as ast

# Path to single hdf5 waterfall file
filename = "/home/lekness/swarmSETI/055917_000007389_Jupiter-waterfall.hdf5"

hf = h5py.File(filename,'r')

#start with an observation1 key which has several observation headers
obs_hdrs = hf['Observation1'].attrs.keys()
#headers: [u'ObservationName', u'TargetName', u'tInt', u'tInt_Unit', u'LFFT', u'nChan',
# u'RBW', u'RBW_Units', u'RA', u'RA_Units', u'Dec', u'Dec_Units', u'Epoch', u'TrackingMode', 
# u'ARX_Filter', u'ARX_Gain1', u'ARX_Gain2', u'ARX_GainS', u'Beam', u'DRX_Gain', u'sampleRate', u'sampleRate_Units']

# time has format and scale headers
time_hdr = hf['Observation1/time'].attrs.keys()

# makes dataset of the times
time = hf['Observation1/time']

# get's start time? 
start_t = time[0][0] + time[0][1]

# convert start time to MJD
start_t_mjd = ast.jd_to_mjd(ast.unix_to_utcjd(start_t))


# each observation has Tuning1, tuning2, and time keys
# each tuning has the following keys: [u'I, u'Q', u'Saturation', u'U', u'V', u'freq']
# time has 2 attributes but tuning doesn't have any attributes

# make set of the I values for each tuning
id_set1 = hf['Observation1/Tuning1/I']
id_set2 = hf['Observation1/Tuning2/I']

# dataset attributes are [u'axis0', u'axis1']
# frequencies will be different for each tuning so need to be accessed separately
# also convert to MHz
freq_1 = (hf['Observation1/Tuning1/freq'][()])/1e+6
freq_2 = (hf['Observation1/Tuning2/freq'][()])/1e+6 

# print the keys: [u'I', u'Q', u'Saturation', u'U', u'V', u'freq']
print(hf['Observation1/Tuning1'].keys())

# now write them into different files each with different tunings

# cut off 10% of the data from the beginning and end
# from claude: 
def write_tuning(hf, filename, outfile_suffix, idset, freq_tun, t0_mjd):
    """Write a single tuning to its own filterbank HDF5 file."""
    outfile = os.path.splitext(os.path.basename(filename))[0] + outfile_suffix
    print(outfile)

    hf_out = h5py.File(outfile, "w")
    hf_out.attrs['CLASS'] = 'FILTERBANK'
    hf_out.attrs['VERSION'] = '1.0'

    dshape = (idset.shape[0], 1, idset.shape[1])
    print(dshape)

    dset = hf_out.create_dataset("data", shape=dshape, dtype=idset.dtype)
    dset_mask = hf_out.create_dataset("mask", shape=dshape, dtype="uint8")

    dset.dims[2].label = b"frequency"
    dset.dims[1].label = b"feed_id"
    dset.dims[0].label = b"time"

    dset_mask.dims[2].label = b"frequency"
    dset_mask.dims[1].label = b"feed_id"
    dset_mask.dims[0].label = b"time"

    dset[:,0,:] = idset
    dset_mask[:,0,:] = np.zeros(idset.shape, dtype='uint8')


    # add attributes to the dataset
    dset.attrs['machine_id'] = 0
    dset.attrs['telescope_id'] = -1
    dset.attrs['src_raj'] = hf['Observation1'].attrs['RA']
    dset.attrs['src_dej'] = hf['Observation1'].attrs['Dec']
    dset.attrs['az_start'] = 0
    dset.attrs['za_start'] = 0
    dset.attrs['data_type'] = 1
    dset.attrs['fch1'] = freq_tun[0]
    dset.attrs['foff'] = freq_tun[1] - freq_tun[0]
    dset.attrs['nchans'] = hf['Observation1'].attrs['nChan']
    dset.attrs['nbeams'] = 1
    dset.attrs['ibeam'] = -1
    dset.attrs['nbits'] = 32
    dset.attrs['tstart'] = start_t_mjd
    dset.attrs['tsamp'] = hf['Observation1'].attrs['tInt']
    dset.attrs['nifs'] = 1
    dset.attrs['source_name'] = hf['Observation1'].attrs['TargetName']
    dset.attrs['rawdatafile'] = os.path.basename(filename).split('-')[0]

    hf_out.close()


# Call it for each tuning and add extensions to each filename
# format: id_set, freq_tun, suffix
tunings = [
    (id_set1, freq_1, "_tun1.h5"),
    (id_set2, freq_2, "_tun2.h5"),
]

for idset, freq_tun, suffix in tunings:
    write_tuning(hf, filename, suffix, idset, freq_tun, start_t_mjd)

hf.close()