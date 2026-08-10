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
 `python3 first_half_pipeline/pipeline.py `
 `--tar-path '/data/network/recent_data/lekness/DT006_260605_0600_0001/DT006_*.tgz' `
` --drx-path '/data/network/recent_data/lekness/DT006_260605_0600_0001/'`
` --avg 3.0`
` --length 8388608`

The second call will create a bandpass profile for each tuning file, then chunk and run BLISS, and then plot cross-check across multiple stations, and finally plot "stamps" of each hit matched at multiple stations. This call must include the direct paths to each of the tuning files (up to 4 or 6), along with chosen tolerances for the anti-coincidence test between frequency and drift-rates. The defaults are 10 Hz across and 0.8 Hz/s. These tolerances produce an average of 1 hit matched between two stations per observation. Other arguments can be tweaked, such as the stamp width (number of frequency channels around the hit), chunk size (cut of tuning file that runs through bliss), min_overlap (overlap across chunks that run through bliss), and output directory. This call first looks if there is already an existing bandpass profile for each tuning file, this way re-runs don't have to produce another as the bandpass generation takes ~10 minutes per tuning file. 

Second call example: \
 `python3 second_half_pipeline/run_pipeline.py \
     frb_set_two/059613_002172869-LWA-SV_tun1.h5 frb_set_two/059613_002172869-LWA-SV_tun2.h5 \
     frb_set_two/059613_002567249-LWA1_tun1.h5   frb_set_two/059613_002567249-LWA1_tun1.h5 \
     --freq_tol 10 --drift_tol 0.8 --width 524 --outdir results_frb_set_three`

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
