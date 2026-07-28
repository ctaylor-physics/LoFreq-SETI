'''
Takes csv files of bliss hits from each of stations for a particular tuning and determines if there are common hits across all 
stations with a certain tolerance: (start with:
    ±10 Hz in frequency 
    ±0.8 Hz/s in drift rate)

Inputs:
    A folder of .csv files that are the hits for each station.
    You may supply EITHER all three of --one/--na/--sv, OR just two of them.

Outputs (depends on how many stations were supplied):
    If 3 stations supplied:
        - hits common across all 3
        - hits common between each pair (NA-SV, NA-1, SV-1)
    If 2 stations supplied:
        - hits common between that pair only

'''

import numpy as np
import pandas as pd
import os
import argparse


# ── CLI arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Anti-coincidence check across LWA stations (2 or 3).')
parser.add_argument('--one',       default=None,  help='Path to LWA-1 hits CSV (optional)')
parser.add_argument('--na',        default=None,  help='Path to LWA-NA hits CSV (optional)')
parser.add_argument('--sv',        default=None,  help='Path to LWA-SV hits CSV (optional)')
parser.add_argument('--freq_tol',  type=float, default=10.0,
                    help='Frequency tolerance in Hz (default: 10 Hz)')
parser.add_argument('--drift_tol', type=float, default=0.8,
                    help='Drift rate tolerance in Hz/s (default: 0.8 Hz/s)')
parser.add_argument('--outdir',    default='.', help='Output directory (default: current dir)')
args = parser.parse_args()

# --- figure out which stations were supplied -----------------------------------
stations = {}
if args.one is not None:
    stations['ONE'] = args.one
if args.na is not None:
    stations['NA'] = args.na
if args.sv is not None:
    stations['SV'] = args.sv

if len(stations) < 2:
    parser.error("You must supply at least two of --one, --na, --sv.")

freq_tolerance  = args.freq_tol  / 1e6   # convert Hz → MHz to match the CSV column units
drift_tolerance     = args.drift_tol          # Hz/s, matches Drift_Rate column directly

# --- load CSVs ------------------------
expected_data_cols = [
        "index", "Drift_Rate", "SNR",
        "Uncorrected_Frequency", "Corrected_Frequency",
        "Index", "freq_start", "freq_end",
        "SEFD_freq", "Coarse_Channel_Number","channel two", "Full_number_of_hits",
    ]


# read in all of the csv files as dataframes and adjust the columns as necessary
def load_hits(path): 
    df = pd.read_csv(path)
    df.columns = expected_data_cols
    return df

# load only the stations that were actually passed in
data = {name: load_hits(path) for name, path in stations.items()}

print(f"Stations supplied: {', '.join(data.keys())}")


def check_between_two_stations(station_1_data, station_2_data, freq_tol, drift_tol):
    '''
    Find hits that are common between two stations within the given tolerances.
 
    A hit from station 1 is considered a match for a hit from station 2 when BOTH:
        |freq_1 - freq_2|   <= freq_tol_MHz   (in MHz)
        |drift_1 - drift_2| <= drift_tol       (in Hz/s)
 
    Parameters
    ----------
    station_1_data : pd.DataFrame
        Hits table for station 1.
    station_2_data : pd.DataFrame
        Hits table for station 2.
    freq_tol_MHz : float
        Frequency tolerance in MHz.
    drift_tol : float
        Drift rate tolerance in Hz/s.
 
    Returns
    -------
    matched_s1 : pd.DataFrame
        Rows from station_1_data that have at least one match in station_2_data.
    matched_s2 : pd.DataFrame
        Rows from station_2_data that have at least one match in station_1_data.
    '''
    # Pull out numpy arrays for fast vectorised comparison
    freq1  = station_1_data['Corrected_Frequency'].to_numpy()   # MHz
    drift1 = station_1_data['Drift_Rate'].to_numpy()             # Hz/s
 
    freq2  = station_2_data['Corrected_Frequency'].to_numpy()
    drift2 = station_2_data['Drift_Rate'].to_numpy()
 
    # Boolean mask: which rows in station 1 match at least one row in station 2
    # Shape of the broadcast comparison: (len_s1, len_s2)
    freq_match  = np.abs(freq1[:, None]  - freq2[None, :]) <= freq_tol
    drift_match = np.abs(drift1[:, None] - drift2[None, :]) <= drift_tol

    both_match = freq_match & drift_match   # (len_s1, len_s2) bool array
 
    # A station-1 row matches if ANY station-2 row satisfies both conditions
    s1_has_match = both_match.any(axis=1)   # shape (len_s1,)
    s2_has_match = both_match.any(axis=0)   # shape (len_s2,)
 
    matched_s1 = station_1_data[s1_has_match].reset_index(drop=True)
    matched_s2 = station_2_data[s2_has_match].reset_index(drop=True)
 
    return matched_s1, matched_s2

# ---- Match hits between stations and make sure all indices align --------
def match_triple_coincidence(data_na, data_sv, data_one, freq_tol, drift_tol):
    """
    Explicitly build row-aligned triple-coincidence matches across
    NA, SV, and LWA-1, using NA as the reference station.

    Returns three row-aligned DataFrames (same length, same order):
    rows_na.iloc[i], rows_sv.iloc[i], rows_one.iloc[i] all refer to
    the SAME physical hit.
    """
    freq_na, drift_na   = data_na['Corrected_Frequency'].to_numpy(), data_na['Drift_Rate'].to_numpy()
    freq_sv, drift_sv   = data_sv['Corrected_Frequency'].to_numpy(), data_sv['Drift_Rate'].to_numpy()
    freq_one, drift_one = data_one['Corrected_Frequency'].to_numpy(), data_one['Drift_Rate'].to_numpy()

    na_sv_match  = (np.abs(freq_na[:, None]  - freq_sv[None, :])  <= freq_tol) & \
                   (np.abs(drift_na[:, None] - drift_sv[None, :]) <= drift_tol)
    na_one_match = (np.abs(freq_na[:, None]  - freq_one[None, :]) <= freq_tol) & \
                   (np.abs(drift_na[:, None] - drift_one[None, :]) <= drift_tol)

    rows_na, rows_sv, rows_one = [], [], []

    for i in range(len(data_na)):
        sv_idxs  = np.where(na_sv_match[i])[0]
        one_idxs = np.where(na_one_match[i])[0]
        if len(sv_idxs) == 0 or len(one_idxs) == 0:
            continue
        # If there are multiple candidate matches (rare), emit one row
        # per combination so nothing gets silently dropped or merged.
        for j in sv_idxs:
            for k in one_idxs:
                rows_na.append(data_na.iloc[i])
                rows_sv.append(data_sv.iloc[j])
                rows_one.append(data_one.iloc[k])

    all_three_na  = pd.DataFrame(rows_na).reset_index(drop=True)
    all_three_sv  = pd.DataFrame(rows_sv).reset_index(drop=True)
    all_three_one = pd.DataFrame(rows_one).reset_index(drop=True)

    return all_three_na, all_three_sv, all_three_one


# ── Save helper ─────────────────────────────────────────────────────────────
os.makedirs(args.outdir, exist_ok=True)

def save(df, name):
    path = os.path.join(args.outdir, name)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df):>5} hits → {path}")


station_names = list(data.keys())

# ── Case 1: exactly two stations supplied ───────────────────────────────────
if len(station_names) == 2:
    name1, name2 = station_names
    print(f"Checking {name1} ↔ {name2} …")
    matched_1, matched_2 = check_between_two_stations(
        data[name1], data[name2], freq_tolerance, drift_tolerance
    )

    print("\nWriting output files:")
    save(matched_1, f'hits_{name1}_{name2}_{name1}.csv')
    save(matched_2, f'hits_{name1}_{name2}_{name2}.csv')

# ── Case 2: all three stations supplied ─────────────────────────────────────
elif len(station_names) == 3:
    data_one = data['ONE']
    data_na  = data['NA']
    data_sv  = data['SV']

    # --- Pairing checks between each of the stations ----
    print("Checking NA ↔ SV …")
    na_in_na_and_sv, sv_in_na_and_sv   = check_between_two_stations(data_na,  data_sv,  freq_tolerance, drift_tolerance)

    print("Checking NA ↔ LWA-1 …")
    na_in_na_and_one, one_in_na_and_one = check_between_two_stations(data_na,  data_one, freq_tolerance, drift_tolerance)

    print("Checking SV ↔ LWA-1 …")
    sv_in_sv_and_one, one_in_sv_and_one = check_between_two_stations(data_sv,  data_one, freq_tolerance, drift_tolerance)

    # --- Triple coincidence -------------------------------
    #  hit appears in all three when it is already in the NA↔SV matched set AND
    #  also matches something in LWA-1.
    print("Checking NA ↔ SV ↔ LWA-1 (row-aligned)…")
    all_three_na, all_three_sv, all_three_one = match_triple_coincidence(data_na, data_sv, data_one, freq_tolerance, drift_tolerance)
   
    print("\nWriting output files:")
    save(all_three_na,  'hits_all_three_NA.csv')
    save(all_three_sv,  'hits_all_three_SV.csv')
    save(all_three_one, 'hits_all_three_ONE.csv')
    save(na_in_na_and_sv,    'hits_na_sv_NA.csv')
    save(sv_in_na_and_sv,    'hits_na_sv_SV.csv')
    save(na_in_na_and_one,   'hits_na_one_NA.csv')
    save(one_in_na_and_one,  'hits_na_one_ONE.csv')
    save(sv_in_sv_and_one,   'hits_sv_one_SV.csv')
    save(one_in_sv_and_one,  'hits_sv_one_ONE.csv')

print("\nDone.")


# example usage:
'''
# two stations
python anti_coincidence.py --one results_frb_4/tun1/hits/059658_002623753-LWA-SV_tun1_bliss_hits.csv --sv results_frb_4/tun1/hits/059658_003090021-LWA1_tun1_bliss_hits.csv --freq_tol 20 --drift_tol 1.0 --outdir ./results

# three stations (same as before)
python3 anti_coincidence.py --one results_day_obs/tun2/hits/061196_001478830-LWA1_tun2_bliss_hits.csv --na results_day_obs/tun2/hits/061196_008848197-LWA-NA_tun2_bliss_hits.csv --sv results_day_obs/tun2/hits/061196_000885206-LWA-SV_tun2_bliss_hits.csv --outdir ./anti_aligned_test

'''
