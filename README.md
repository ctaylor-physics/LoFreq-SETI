# LoFreq-SETI
Project to conduct SETI searches using observations with the LWA Swarm.

## Project Description
The LoFreq SETI Project is a tool that searches Long Wavelength Array (LWA) data for signs of extraterrestrial intelligence. Technosignatures are narrow-band radio emissions occurring with a specific drift rate due to the Doppler shift between the Earth's motion and an extraterrestrial emitter. The purpose of this project is to expand the limits of SETI research by using the LWA, which takes data at frequencies lower than typical SETI searches (LWA searches 10-88 MHz, while SETI usually searches the "water hole" at 1.42-1.66 GHz). Additionally, since the LWA Swarm is configured with multiple "stations", an anti-coincidence check can be done to ensure that candidate signals are received by multiple stations at the same time. Signal candidates (aka "hits") are plotted at the end of the pipeline for visual inspection and comparisons.

## Pipeline Layout
The pipeline is split into two sections.
<img width="3200" height="2400" alt="2" src="https://github.com/user-attachments/assets/cc6663a3-13c8-4362-a603-40e7edd043f7" />
The left column is the first half of the pipeline and the right column is the second half. After the first half, there is a break for potential inspection and reorganizing of the data if necessary. The second half of the pipeline is called separately. This provides freedom for comparing multiple station data in case one station did not have adequate tuning file quality. 


## Instructions
Each half of the pipeline requires a command line call. The first call will produce 2 tuning files for each stations' observations. This call requires a path to the tarballs and the raw data files, along with resolution preferences for the time and frequency channelizations. For high frequency resolution (which is necessary for technosignature searches), use an averaging length of 3.0 s and an FFT length of 8388608. These variables are adjustable, therefore changes can be made for coarser frequency resolution if a shorter processing time is desired, for example. 

First call example: \
 (run like this for all stations on the same date (e.g. DT006_260605_0600_0001)):\
 `python3 first_half_pipeline/pipeline.py `\
 `--tar-path '/data/network/recent_data/lekness/DT006_260605_0600_0001/DT006_*.tgz' `\
` --drx-path '/data/network/recent_data/lekness/DT006_260605_0600_0001/'`\
` --avg 3.0`\
` --length 8388608`\
` --num-coarse 32 --edge-coarse 3`

### Full-grid first-half output

The first half now retains the complete FFT frequency grid instead of trimming
10% from each end. Both entry points (`pipeline.py` and `upchannelize.py`) accept:

- `--num-coarse` (default **32**): total coarse channels across the full tuning.
  This must be positive and divide the FFT length exactly.
- `--edge-coarse` (default **3**): coarse channels excluded **at each edge**.
  This may be zero and must leave at least one searchable coarse channel.

For `--length 8388608 --num-coarse 32 --edge-coarse 3`, each coarse channel
contains 262144 fine channels. The first and last 786432 fine channels are set
to **-1.0**, removing 9.375% per edge from the usable band. The interior data
are unchanged. `data` and `mask` retain shape `(time, 1, 8388608)`; the `uint8`
mask is **1** at the excluded edges and **0** in the valid interior. Other FFT
lengths can use other layouts, e.g. `--length 4096 --num-coarse 8 --edge-coarse 1`.

`data.attrs` records the layout for subsequent bandpass fitting and searches:

| Attribute | Meaning |
| --- | --- |
| `coarse_layout_version` | Layout metadata version (1) |
| `num_coarse` | Total coarse-channel count |
| `edge_coarse` | Number excluded at each edge |
| `fine_channels_per_coarse` | Actual fine-channel count divided by `num_coarse` |
| `valid_channel_start` | First valid fine-channel index, inclusive |
| `valid_channel_stop` | End of valid fine-channel interval, exclusive |
| `edge_mask_value` | Sentinel value (-1.0) |

Indices are zero-based and follow dataset order, including for descending
frequency axes. `fch1`, `foff`, and `nchans` describe the full grid; the valid
frequency interval can be derived from these and the stored index bounds.
Both actual tuning shapes are validated before writing the output files.

To run from another directory, use an absolute script path, or export the repo
root on `PYTHONPATH` and invoke `python3 -m first_half_pipeline.pipeline` with the
same arguments. Use your patched virtual environment's Python and existing LSL
environment settings for legacy data. Output tuning files are written in the
current working directory.

### Fit and apply the bandpass

Run the updated fitting stage independently on each new full-grid tuning file:

```bash
/path/to/venv/bin/python3 /path/to/LoFreq-SETI/second_half_pipeline/lwa_bliss_bp_gen.py \
  /path/to/observation-LWA1_tun1.h5
```

The fitter reads and validates the layout recorded by the first half; it does
not take new coarse-channel settings. It computes the instrumental response and
fits the sampled residual spectrum **only within the valid interval**. Both the
residual and final smoothing stay within that interval, and all normalizations
exclude the edge sentinels. The instrumental response is scaled by its mean,
without subtracting its minimum. Nonfinite/nonpositive responses fail explicitly
instead of being replaced by tiny denominators.

The **final smoothed, mean-one float32 profile** is used for correction and saved
as `bandpass_model` in the same HDF5 file. It has one value per original fine
channel, with neutral values of **1.0** outside the valid interval. The exact same
array is exported as `<basename>_bpmodel.f32` in the current working directory;
use `--output /path/to/profile.f32` to choose another location. No further
smoothing is performed on export.

Once all corrected blocks have been validated, the file contains:

| Dataset | Contents |
| --- | --- |
| `uncorrected` | Original first-half data, retained for inspection and refitting |
| `data` | Flattened interior, with excluded edges still equal to -1.0 |
| `mask` | Original mask, unchanged |
| `bandpass_model` | Final profile actually applied to the data |

The corrected `data` preserves filterbank attributes, layout metadata, and axis
labels. Correction version, source dataset, sampled-row count, instrumental-model
usage, and actual residual/final smoothing windows and orders are recorded on the
data and model. The model also records the original coarse layout.

A normal rerun reuses a completed correction and recreates the profile export.
Use `--force` to refit **uncorrected**, including when changing fitting parameters;
it never fits already-flattened data. An interrupted block write is detected and
requires `--force` to restart. A completed staged correction interrupted during
promotion is finished on the next invocation. Staging uses HDF5 hard links to
preserve the original datasets during promotion; it is not a guarantee against
filesystem corruption or power-loss damage to the HDF5 file itself.

Defaults are `--sample-rows 32`, `--row-block 16`, `--channel-block 131072`,
`--window-size 41` for the residual fit, and an odd window near the square root
of the valid-channel count for final smoothing (`--final-window` overrides it).
Smoothing windows/orders are reduced for short valid intervals. Fitting samples
and correction are processed in bounded frequency blocks, with CuPy used when
an accessible GPU is available and NumPy otherwise. Corrected data are written
without compression to avoid another compression bottleneck; retaining original
and corrected data requires space for both datasets. Failed/repeated refits can
leave HDF5 free space; deleting datasets does not necessarily shrink the file.

**Integration status:** the first half and standalone fitter now support this
layout. The next step is replacing the old chunk-and-BLISS workflow with direct
coarse-channel processing, including explicit edge exclusion and BLISS
preprocessing control. Do not run the existing end-to-end second-half runner on
these files yet. The legacy chunker rejects already-corrected data to prevent
double bandpass correction. The mask and -1.0 values do not make BLISS skip the
excluded channels automatically. Old files without layout metadata and legacy
files containing a `corrected` dataset are rejected; start from updated
first-half output.

Run the synthetic first-half and fitting checks with:

```bash
python3 -m unittest discover -s tests -v
```

The second call will create a bandpass profile for each tuning file, then chunk and run BLISS, and then plot cross-check across multiple stations, and finally plot "stamps" of each hit matched at multiple stations. This call must include the direct paths to each of the tuning files (up to 4 or 6), along with chosen tolerances for the anti-coincidence test between frequency and drift-rates. The defaults are 10 Hz across and 0.8 Hz/s. These tolerances produce an average of 1 hit matched between two stations per observation. Other arguments can be tweaked, such as the stamp width (number of frequency channels around the hit), chunk size (cut of tuning file that runs through bliss), min_overlap (overlap across chunks that run through bliss), and output directory. This call first looks if there is already an existing bandpass profile for each tuning file, this way re-runs don't have to produce another as the bandpass generation takes ~10 minutes per tuning file. 

Second call example: \
 `python3 second_half_pipeline/run_pipeline.py` \
 `    frb_set_two/059613_002172869-LWA-SV_tun1.h5 frb_set_two/059613_002172869-LWA-SV_tun2.h5` \
`     frb_set_two/059613_002567249-LWA1_tun1.h5   frb_set_two/059613_002567249-LWA1_tun1.h5` \
`     --freq_tol 10 --drift_tol 0.8 --width 524 --outdir results_frb_set_three`

## Usage on older data
While this codebase has been tested on both old and new data, it was created primarily for usage on new data from the LWA. For archival data, patches must be made. Particularly, the lsl package must run on a different version for older data, so a virtual environment should be created. The details of the virtual environment are listed in "venv-legacy-lsl.md". This virtual environment was created specifically for data recorded in 2022 using an lsl version of 3.0.8. Additional or fewer patches may be necessary for archival data of a different era. 

## Package Dependencies
The requirements for this codebase are the same as those for the lsl package:
- python >= 3.8
- numpy >= 1.7
- scipy >= 0.19
- astropy >= 5.2
- pyephem >= 3.7.5.3
- aipy >= 3.0.1


### This pipeline was built for UNM REU project 2026
