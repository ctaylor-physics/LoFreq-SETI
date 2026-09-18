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

Savitzky–Golay fits can undershoot below zero around strong narrow spectral
features. Both smoothing stages keep the linear fit when it is finite and
positive; otherwise they refit the positive input in log space, bounded to the
observed input range before normalization. The selected methods are logged and
stored as `residual_smoothing_method` and `final_smoothing_method` on `data` and
`bandpass_model`. Invalid input profiles still fail, with counts and affected
fine-channel indices, rather than being silently floored or interpolated.

#### Optional clipping of the fitting spectrum

Clipping is **off by default**. Enable it for comparison tests with `--fit-clip`:

```bash
python3 -m second_half_pipeline.lwa_bliss_bp_gen tuning.h5 --fit-clip
# An already-corrected file must be refitted from uncorrected:
python3 -m second_half_pipeline.lwa_bliss_bp_gen tuning.h5 --force --fit-clip \\
  --fit-clip-sigma 10 --fit-clip-window 101 --fit-clip-max-width 8
```

Clipping acts once on the time-median spectrum after instrumental-response
division, before either smoothing pass. A running median estimates the local
baseline; 1.4826 times the running median of absolute residuals estimates robust
scatter. Only **positive** deviations above `--fit-clip-sigma` are candidates.
Contiguous candidate runs of at most `--fit-clip-max-width` fine channels are
interpolated between their unflagged neighbors. Longer runs and runs touching
the valid interval's endpoints are left unchanged. The window must be odd, at
least three channels, and no longer than the valid interval; the maximum run
width must be positive and less than half the window. These widths are in fine
channels, so their frequency spans depend on the FFT length.

The raw observations and `mask` are never edited by clipping. Settings, candidate
counts, skipped-run counts, and rejected-bin counts are stored on `data` and
`bandpass_model`. `bandpass_fit_clipped_channels` stores the zero-based **full-grid**
indices interpolated in the fitting spectrum (empty when disabled). Changing
clipping settings on a completed file requires `--force`; omitting `--fit-clip`
with `--force` restores fitting without clipping, always from original data.

The enabled starting defaults are **10 robust sigma, a 101-channel window, and
an eight-channel maximum run**. On the supplied LWA1 example (6,815,744 valid
channels), trials at 10, 15, 20, and 30 sigma rejected 166, 104, 77, and 52 bins,
respectively. At 10 sigma only 0.00244% of fitting bins were replaced, both
smoothing passes stayed positive without a log fallback, and the maximum model
value dropped from 4.055 to 3.270. After median-normalizing profiles, 95% of
channels changed by less than 0.0193%. This is a conservative starting point
from one example, not a universal calibration: broader or less extreme features
can remain in the model, and signal-recovery sensitivity still needs testing.

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
| `bandpass_fit_clipped_channels` | Fine-channel indices interpolated only in the fitting spectrum |

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

The second-half runner now fits each full-grid file, searches it directly with
BLISS, matches hits across stations, and plots coincidence stamps. It does not
create HDF5 chunks. Old trimmed files without layout metadata are rejected;
start from updated first-half output.

```bash
python3 -m second_half_pipeline.run_pipeline \
    observation-LWA1_tun1.h5 observation-LWA1_tun2.h5 \
    observation-LWA-SV_tun1.h5 observation-LWA-SV_tun2.h5 \
    --bliss-executable /path/to/bliss_find_hits \
    --fit-clip --freq_tol 10 --drift_tol 0.8 --width 524 \
    --outdir pipeline_results
```

Supply both tunings for either two or three stations. Optional fitting-spectrum
clipping remains disabled unless `--fit-clip` is supplied; its conservative
settings are sigma 10, window 101, maximum run width 8. To change clipping on an
already-corrected file, use `--force` to refit from `uncorrected`. Completed
corrections are detected inside HDF5, not from the existence of an exported
`.f32` file.

To search a single file that has already been fitted:

```bash
python3 -m second_half_pipeline.run_bliss --h5 tuning.h5 \
    --bliss-executable /path/to/bliss_find_hits --outdir hits
```

Both commands accept `--device` (e.g. `cpu` or `cuda:0`; otherwise BLISS chooses),
`--snr` (10), `--min-drift` (-3 Hz/s), `--max-drift` (3 Hz/s),
`--drift-step` (1), and `--distance` (7). The executable defaults to
`BLISS_FIND_HITS` if set, otherwise `bliss_find_hits` on PATH. From another
working directory, put the repository root on `PYTHONPATH` for `-m`, or invoke
these scripts by their absolute filesystem paths.

The coarse-channel width and excluded edges come from the HDF5 metadata:
BLISS receives `--nchan-per-coarse`, `--coarse-channel`, and `--number-coarse`.
For 32 coarse channels with three excluded at each edge, the search covers
channels 3 through 28 (zero-based). The mask and sentinel values alone do not
make BLISS skip edges. A generated unity `.f32` profile is passed with `-e`
because omitting it invokes BLISS's default rolloff processing; the real
bandpass has already been applied to `data`. Native coarse-channel boundaries
have no overlapping windows, so boundary-crossing hits can differ from the
legacy overlapping-chunk search. The old `--nchan`, `--chunksize`, and
`--min_overlap` runner flags no longer apply.

Each search retains its original `.dat`, canonical hits `.csv`, unity profile,
BLISS log, and a JSON search manifest. Reuse requires matching input and
executable paths/sizes/modification times, search settings, and a verified
`.dat` hash. The CSV is regenerated even on reuse; coincidence tables and plots
are regenerated on every pipeline run. `--force` refits and searches again.

BLISS currently writes 12 data fields under an 11-field `.dat` header. The
converter uses the actual serialized order, inserting the missing `SEFD` field
before `SEFD_freq` and naming the last field `Bin_Width` (the upstream header
calls it `Full_number_of_hits`). `Index` is a fine-channel index **within** its
coarse channel; plots use the frequency in MHz instead. CSVs have 12 named
columns with no extra pandas index. Ambiguous legacy CSVs are rejected rather
than guessed. Regenerate them from their original `.dat` files:

```bash
python3 -m second_half_pipeline.hits_io old_hits.dat --output repaired_hits.csv
```

Run the first-half, fitting, parser, and direct-runner regression checks with:

```bash
python3 -m unittest discover -s tests -v
```

The direct-runner tests use a fake BLISS executable to exercise orchestration,
caching, and downstream plotting. Scientific detection performance should be
checked with the installed BLISS build on the compute cluster.

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
