At this early stage, postprocessing deals with attempting to recycle candidate hits for re-correlation at high frequency resolution.

This would be done by identifying signals of interest using the multistation coincidence filter, creating a narrow correlation window to process at high resolution, then convert the correlated (and possibly calibrated) visibilities into a suitable format for BLISS to search. Hopefully signals that are technosignatures would also appear as hits in this reanalysis with BLISS.

Scripts: 
uvfits_to_filterbank.py
 - Converts a uvfits output file of the ELWA Correlator into an equivalent filterbank style H5 file for BLISS. 
