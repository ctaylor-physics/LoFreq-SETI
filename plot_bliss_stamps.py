"""
plot_bliss_stamps.py

Make postage stamp waterfall plots for each hit in a bliss output CSV.
Reads directly from the original full .h5 file.

Usage:
    python plot_bliss_stamps.py <hits.csv> <waterfall.h5> [--width 524] [--outdir stamps]

Outputs:
    - One PNG per hit in <outdir>/
    - One multi-page PDF of all hits: <outdir>/all_stamps.pdf
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def freq_to_chan(freq_mhz, fch1, foff):
    """Convert a frequency in MHz to the nearest channel index."""
    return int(round((freq_mhz - fch1) / foff))


def load_hits(csv_path):
    """
    Load the bliss hits CSV. The file has a leading pandas index column
    so we add index column = 0 and re-align all the columns to the expected names. 
    Also adds a sequential hit number column.
    """
    raw = pd.read_csv(csv_path, header=0)

    expected_data_cols = [
        "index", "Drift_Rate", "SNR",
        "Uncorrected_Frequency", "Corrected_Frequency",
        "Index", "freq_start", "freq_end",
        "SEFD_freq", "Coarse_Channel_Number","channel two", "Full_number_of_hits",
    ]

    raw.columns = expected_data_cols  # ensure consistent column names
    # print(f"Raw columns: {list(raw.columns)}") # uncomment for debugging
    # print(f"First row values: {raw.iloc[0].tolist()}")


    df = raw[expected_data_cols].copy()
    df.columns = expected_data_cols  # ensure consistent column names

    # Add a sequential hit number based on row position (Top_Hit_# resets per chunk)
    df["hit_num"] = np.arange(1, len(df) + 1)

    # Pick the MHz frequency column: whichever of Uncorrected/Corrected is < 500
    corr = df["Corrected_Frequency"].values
    uncorr = df["Uncorrected_Frequency"].values # these should be identical at this float depth

    if corr.max() < 500:
        df["_hit_freq_mhz"] = df["Corrected_Frequency"]
        print(f"Using Corrected_Frequency. Sample: {corr[:5]}")
    else:
        df["_hit_freq_mhz"] = df["Uncorrected_Frequency"]
        print(f"Corrected_Frequency looks like indices; using Uncorrected_Frequency. Sample: {uncorr[:5]}")

    return df.reset_index(drop=True)



def make_stamp(h5file, fch1, foff, tsamp, hit_freq, half_width):
    '''
    Extract a postage stamp centered on hit_freq (MHz).
    half_width is in channels.
    Returns stamp (time x freq), freqs (MHz), times (s).
    '''
    nchan_total = h5file['data'].shape[2]
    ntime       = h5file['data'].shape[0]

    freq_min_file = min(fch1, fch1 + foff * (nchan_total - 1))
    freq_max_file = max(fch1, fch1 + foff * (nchan_total - 1))

    if not (freq_min_file <= hit_freq <= freq_max_file):
        raise ValueError(
            f"Hit frequency {hit_freq:.6f} MHz outside file range "
            f"[{freq_min_file:.4f}, {freq_max_file:.4f}] MHz"
        )

    center_chan = freq_to_chan(hit_freq, fch1, foff)
    center_chan = max(0, min(nchan_total - 1, center_chan))

    c_start = max(0, center_chan - half_width)
    c_stop  = min(nchan_total, center_chan + half_width)

    # data shape: (time, feed_id, freq) — squeeze out feed_id
    stamp = h5file['data'][:, 0, c_start:c_stop].astype(np.float32)
    freqs = fch1 + foff * np.arange(c_start, c_stop)
    times = tsamp * np.arange(ntime)

    return stamp, freqs, times



def make_stamps(hits_csv, h5_path, stamp_width=524, stamp_dir='stamps'):
    """Generate PNG + PDF postage stamps for every hit in hits_csv."""
    os.makedirs(stamp_dir, exist_ok=True)
    half_width = stamp_width // 2
 
    hits = load_hits(hits_csv)
    print(f"Loaded {len(hits)} hits")
 
    with h5py.File(h5_path, 'r') as hf:
        attrs = dict(hf['data'].attrs)
        fch1  = float(attrs['fch1'])
        foff  = float(attrs['foff'])
        tsamp = float(attrs['tsamp'])
 
        nchan_total = hf['data'].shape[2]
        freq_end    = fch1 + foff * (nchan_total - 1)
        print(f"H5: fch1={fch1:.6f} MHz  foff={foff:.8f} MHz  tsamp={tsamp:.4f} s")
        print(f"File freq range: {min(fch1,freq_end):.4f} - {max(fch1,freq_end):.4f} MHz")
 
        if hits["_hit_freq_mhz"].isna().any():
            print("Converting Index channel numbers to MHz using fch1/foff...")
            hits["_hit_freq_mhz"] = fch1 + foff * hits["Index"]
 
        print(f"Hit freq range:  {hits['_hit_freq_mhz'].min():.4f} - {hits['_hit_freq_mhz'].max():.4f} MHz")
 
        pdf_path = os.path.join(stamp_dir, 'all_stamps.pdf')
        with mpdf.PdfPages(pdf_path) as pdf:
            for idx, row in hits.iterrows():
                hit_freq = float(row["_hit_freq_mhz"])
                hit_num  = int(row["hit_num"])
                try:
                    stamp, freqs, times = make_stamp(hf, fch1, foff, tsamp, hit_freq, half_width)
                except Exception as e:
                    print(f"  Skipping hit {hit_num} ({hit_freq:.4f} MHz): {e}")
                    continue
                fig, _ = plot_stamp(stamp, freqs, times, row, hit_freq, hit_num)
                png_name = f"hit_{hit_num:04d}_{hit_freq:.4f}MHz.png"
                fig.savefig(os.path.join(stamp_dir, png_name), dpi=150, bbox_inches='tight')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
                print(f"  [{idx+1}/{len(hits)}] Hit #{hit_num} @ {hit_freq:.6f} MHz  SNR={row['SNR']:.1f}  -> {png_name}")
 
    print(f"Done. PNGs in '{stamp_dir}/'  |  PDF: '{pdf_path}'")


def plot_stamp(stamp, freqs, times, hit_row, hit_freq, hit_num):
    """
    Draw a single waterfall postage stamp.
    - Time increases upward (t=0 at bottom)
    - Drift line clipped to stamp freq bounds
    """
    fig, ax = plt.subplots(figsize=(7, 4))

    vmin = np.nanpercentile(stamp, 1)
    vmax = np.nanpercentile(stamp, 99)

    # origin='lower' + extent with time going low->high = t=0 at bottom
    ext = [freqs[0], freqs[-1], times[0], times[-1]]
    im = ax.imshow(
        stamp,
        # stamp[::-1],          # flip rows so t=0 row is at the bottom visually
        aspect='auto',
        interpolation='nearest',
        origin='lower',
        extent=ext,
        vmin=vmin, vmax=vmax,
        cmap='viridis',
    )

    drift = hit_row['Drift_Rate']   # Hz/s

    # Mark hit frequency
    ax.axvline(hit_freq, color='red', linewidth=0.8, linestyle='--', alpha=0.8, label='Hit freq')

    ax.set_xlabel('Frequency (MHz)', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.tick_params(labelsize=8)
    ax.set_title(
        f"Hit #{hit_num}  |  {hit_freq:.6f} MHz  |  "
        f"SNR {hit_row['SNR']:.1f}  |  Drift {drift:.5f} Hz/s",
        fontsize=11,
    )
    ax.legend(fontsize=7, loc='upper right')
    fig.colorbar(im, ax=ax, label='Power')
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Postage stamp plots for bliss hits.')
    parser.add_argument('csv',  help='Path to bliss hits CSV')
    parser.add_argument('h5',   help='Path to original .h5 waterfall file')
    parser.add_argument('--width',  type=int, default=524,
                        help='Total channel width of each stamp (default: 524)')
    parser.add_argument('--outdir', default='stamps',
                        help='Output directory (default: stamps/)')
    args = parser.parse_args()
    make_stamps(args.csv, args.h5, args.width, args.outdir)
 

if __name__ == '__main__':
    main()

# run like: python3 plot_bliss_stamps.py frb_data/frb_6/results_frb_6/tun1/hits/059704_003060021-LWA-SV_tun1_bliss_hits.csv frb_data/frb_6/059704_003060021-LWA-SV_tun1.h5 --width 524 --outdir stamps