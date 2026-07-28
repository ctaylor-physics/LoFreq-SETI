"""
Retrieves metadata about an observation from its corresponding tarball. 
This metadata is required for upchannelizing the raw data and can also be used to create a csv of the metadata for inspection
"""

import os
import glob
import csv
import tarfile
import pandas as pd
import ephem
from astropy import units as u
from datetime import timezone

from lsl.common import metabundle
from lsl.common.mcs import mjdmpm_to_datetime


def get_metadata(filename, more_info=False):
    """
    Retrieve basic observing information from an LWA observation tarball (.tgz).
    
    Input:
        filename  - path to a .tgz tarball
        more_info - if True, also returns visibility, frequency1, frequency2, filter
    Output:
        pandas DataFrame including: datafile, name, intent, ra2000, dec2000,
                                    startUTC, stopUTC, offset, duration (s)
        if more_info=True, also includes: visibility, frequency1, frequency2, filter
    """
    sdf  = metabundle.get_sdf(filename)
    meta = metabundle.get_session_metadata(filename)
    data_filename = meta[1]['tag']
    offset  = 0
    sources = []

    for o, obs in enumerate(sdf.sessions[0].observations):
        name       = obs.target
        intent     = obs.name
        ra         = ephem.hours(str(obs.ra))
        dec        = ephem.hours(str(obs.dec))
        tStart     = mjdmpm_to_datetime(obs.mjd, obs.mpm, tz=timezone.utc)
        tStop      = mjdmpm_to_datetime(obs.mjd, obs.mpm + obs.dur, tz=timezone.utc)
        duration_s = obs.dur / 1000
        sources.append({
            'datafile':    data_filename,
            'name':        name,
            'intent':      intent,
            'ra2000':      ra,
            'dec2000':     dec,
            'startUTC':    tStart,
            'stopUTC':     tStop,
            'offset':      offset,
            'duration (s)': duration_s,
        })
        offset += duration_s

    sources_df = pd.DataFrame(sources)

    if more_info:
        extras = []
        for o, obs in enumerate(sdf.sessions[0].observations):
            visibility  = obs.target_visibility
            frequency_1 = (obs.freq1 / 1e6) * u.MHz
            frequency_2 = (obs.freq2 / 1e6) * u.MHz
            filt        = obs.filter
            extras.append({
                'visibility':  visibility,
                'frequency1':  frequency_1,
                'frequency2':  frequency_2,
                'filter':      filt,
            })
        extras_df    = pd.DataFrame(extras)
        sources_full = pd.concat([sources_df, extras_df], axis=1)
        return sources_full
    else:
        return sources_df


def get_station(filename, station=None):
    """
    Extract the station name from a tarball's mcs.host file.
    Returns one of: 'lwa1', 'lwasv', 'lwana'
    If station is provided, skip reading the tarball.
    """
    if station is not None:
        return station.lower()
    with tarfile.open(filename) as tar:
        station = tar.extractfile('mcs.host').read().decode().strip().replace('-tp', '')
    return station


def build_csv(tar_glob, output_dir='.', station = None):
    """
    Loop over all tarballs matching tar_glob, determine their station,
    and write one CSV per station to output_dir.

    Output files:
        metadata_lwa1.csv
        metadata_lwa-sv.csv
        metadata_lwa-na.csv
    """
    # Map station names to their CSV filenames
    station_csv_map = {
        'lwa1':  'metadata_lwa1.csv',
        'lwasv': 'metadata_lwa-sv.csv',
        'lwana': 'metadata_lwa-na.csv',
    }

    # Open all three CSV files and write headers
    file_handles = {k: open(os.path.join(output_dir, v), 'w', newline='')
                    for k, v in station_csv_map.items()}
    writers = {k: csv.DictWriter(v, fieldnames=['tarball', 'beamid', 'datafile'])
               for k, v in file_handles.items()}
    for w in writers.values():
        w.writeheader()

    tarballs = sorted(glob.glob(tar_glob))
    print(f"Found {len(tarballs)} tarballs matching: {tar_glob}")

    for t in tarballs:
        station = get_station(t, station = station)
        df      = get_metadata(t)
        writers[station].writerow({
            'tarball':  os.path.basename(t),
            'beamid':   1,
            'datafile': df['datafile'].iloc[0],
        })
        print(f"{os.path.basename(t)} -> {station} -> {df['datafile'].iloc[0]}")

    for f in file_handles.values():
        f.close()

    print(f"\nCSVs written to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build metadata CSVs from LWA tarballs')
    parser.add_argument('--tar-path',   required=True,
                        help='Glob pattern for .tgz files, e.g. "/data/.../DT006_*.tgz"')
    parser.add_argument('--output-dir', default='.',
                        help='Directory to write the CSV files to (default: current directory)')
    args = parser.parse_args()

    build_csv(args.tar_path, args.output_dir)
