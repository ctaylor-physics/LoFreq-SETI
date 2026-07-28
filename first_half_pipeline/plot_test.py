'''
Stand-alone, hardcoded script to visualize a sample of a single tuning file for inspection.
Useful for checking that expected RFI features often around the target frequency are present.

'''

import h5py
import numpy as np
import matplotlib.pyplot as plt

# import file
file = '../frb_data/frb_8/059639_002431255-LWA-SV_tun1.h5' # change to your tuning file path
# add station label for when inspecting multiple stations
station = 'sevilleta'
target_var = 60.0 # make sure the target is within the frequency limits of the tuning file

with h5py.File(file, 'r') as f:
    attrs = dict(f['data'].attrs)
    target = target_var
    area = 274
    idx = int((target - attrs['fch1']) / attrs['foff'])
    sl = slice(idx-area, idx+area)
    data = f['data'][:, 0, sl]   # load only the slice you need, drop feed dimension

freqs = attrs['fch1'] + np.arange(attrs['nchans']) * attrs['foff']
freqs_sl = attrs['fch1'] + np.arange(idx-area, idx+area) * attrs['foff']
times = np.arange(data.shape[0]) * attrs['tsamp'] / 60

plt.figure(figsize=(12, 6))

plt.imshow(data, aspect='auto', origin='lower',
           extent=[freqs_sl[0], freqs_sl[-1], times[0], times[-1]],
           vmin=np.percentile(data, 5), vmax=np.percentile(data, 99))
plt.colorbar(label='Power')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Time (minutes)')
plt.title(attrs['source_name'])
plt.tight_layout()
plt.savefig(f'waterfall_{target_var}MHz_{station}.png', dpi=150, bbox_inches='tight')

print(f"idx: {idx}")
print(f"slice: {sl}")
print(f"data shape: {data.shape}")
print(f"data min: {data.min()}, max: {data.max()}")
print(f"freq range: {freqs_sl[0]:.4f} - {freqs_sl[-1]:.4f} MHz")
