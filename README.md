# LoFreq-SETI
Project to conduct SETI searches using observations with the LWA Swarm.

## Project Description
The LoFreq SETI Project is a tool that searches Long Wavelength Array (LWA) data for signs of extraterrestrial intelligence. Technosignatures are narrow-band radio emissions occurring with a specific drift rate due to the Doppler shift between the Earth's motion and an extraterrestrial emitter. The purpose of this project is to expand the limits of SETI research by using the LWA, which takes data at frequencies lower than typical SETI searches (LWA searches 10-88 MHz, while SETI usually searches the "water hole" at 1.42-1.66 GHz). Additionally, since the LWA Swarm is configured with multiple "stations", an anti-coincidence check can be done to ensure that candidate signals are received by multiple stations at the same time. Signal candidates (aka "hits") are plotted at the end of the pipeline for visual inspection and comparisons.

## Pipeline Layout
The pipeline is split into two sections.
<img width="3200" height="2400" alt="2" src="https://github.com/user-attachments/assets/cc6663a3-13c8-4362-a603-40e7edd043f7" />
The left column is the first half of the pipeline and the right column is the second half. After the first half, there is a break for potential inspection and reorganizing of the data if necessary. The second half of the pipeline is called separately. This provides freedom for comparing multiple station data in case one station did not have adequate tuning file quality. 


## use instructions
audience should be familiar with lwa observations
- preserve raw data for this pipeline

# potentially: 
- package dependancies

look at lsl read me


### This pipeline was built for UNM REU project 2026
