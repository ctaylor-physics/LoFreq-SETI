"""
plot_pair_coincidence.py

For each pairwise-coincidence hit (common between two stations, e.g. from the
2-station mode of anti_coincidence.py), produce a 3-panel figure:

    ┌──────────────┬──────────────┐
    │  Station A   │  Station B   │
    ├──────────────┴──────────────┤
    │             INFO             │
    └──────────────────────────────┘

The INFO panel summarises the hit: frequencies, drift rates, SNRs, and the
Δfreq / Δdrift between the two stations.

Inputs
------
Two matched CSV files produced by anti_coincidence.py in 2-station mode
(e.g. hits_NA_SV_NA.csv, hits_NA_SV_SV.csv) and the corresponding .h5
waterfall file for each station.

Usage
-----
python3 plot_pair_coincidence.py \\
    --csv_a  hits_NA_SV_NA.csv  --h5_a  lwa_na.h5  --label_a LWA-NA \\
    --csv_b  hits_NA_SV_SV.csv  --h5_b  lwa_sv.h5  --label_b LWA-SV \\
    --width 524 --outdir pair_stamps
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── Reused helper (identical logic to plot_bliss_stamps.py) ───────────────────
from plot_bliss_stamps import make_stamp


# ── Matched-CSV loader ──────────────────────────────────────────────────────
#
# NOTE: This is intentionally separate from plot_bliss_stamps.load_hits().
# That loader expects the *raw* bliss hits CSVs, which have a leading pandas
# index column (12 columns total). The matched CSVs written by
# anti_coincidence.py are saved with index=False and only the 11
# expected_data_cols, so we parse those directly here instead.
MATCHED_COLS = [
    "index","Drift_Rate", "SNR",
    "Uncorrected_Frequency", "Corrected_Frequency",
    "Index", "freq_start", "freq_end",
    "SEFD_freq", "Coarse_Channel_Number", "channel two", "Full_number_of_hits",
]


def load_matched_hits(csv_path):
    """Load a matched-hits CSV produced by anti_coincidence.py."""
    df = pd.read_csv(csv_path, header=0)
    df.columns = MATCHED_COLS
    df["hit_num"] = np.arange(1, len(df) + 1)

    corr = df["Corrected_Frequency"].values
    if corr.size == 0:
        print(f"Warning: matched CSV '{csv_path}' is empty - skipping")
        return df
    if corr.max() < 500:
        df["_hit_freq_mhz"] = df["Corrected_Frequency"]
    else:
        df["_hit_freq_mhz"] = df["Uncorrected_Frequency"]

    return df.reset_index(drop=True)


def open_h5(path):
    """Open an .h5 file and return (file_handle, fch1, foff, tsamp)."""
    hf    = h5py.File(path, 'r')
    attrs = dict(hf['data'].attrs)
    fch1  = float(attrs['fch1'])
    foff  = float(attrs['foff'])
    tsamp = float(attrs['tsamp'])
    return hf, fch1, foff, tsamp


# ── Single-quadrant waterfall ──────────────────────────────────────────────────

def draw_waterfall(ax, stamp, freqs, times, label, hit_row):
    """Draw one waterfall panel into ax."""
    vmin = np.nanpercentile(stamp, 1)
    vmax = np.nanpercentile(stamp, 99)
    ext  = [freqs[0], freqs[-1], times[0], times[-1]]

    im = ax.imshow(
        stamp,
        aspect='auto',
        interpolation='nearest',
        origin='lower',
        extent=ext,
        vmin=vmin, vmax=vmax,
        cmap='viridis',
    )
    ax.set_xlabel('Frequency (MHz)', fontsize=11)
    ax.set_ylabel('Time (s)', fontsize=11)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"{label}  |  {hit_row['_hit_freq_mhz']:.6f} MHz  "
        f"|  SNR {hit_row['SNR']:.1f}  |  drift {hit_row['Drift_Rate']:.5f} Hz/s",
        fontsize=9,
    )
    plt.colorbar(im, ax=ax, label='Power', pad=0.02)


# ── Info panel ──────────────────────────────────────────────────────────────

def draw_info(ax, row_a, row_b, label_a, label_b, hit_index, freq_tol_hz, drift_tol):
    """Fill the bottom panel with hit metadata and the pairwise delta."""
    ax.axis('off')

    f_a = row_a['_hit_freq_mhz']; d_a = row_a['Drift_Rate']; snr_a = row_a['SNR']
    f_b = row_b['_hit_freq_mhz']; d_b = row_b['Drift_Rate']; snr_b = row_b['SNR']

    df_a_b = (f_a - f_b) * 1e6   # MHz → Hz
    dd_a_b = d_a - d_b

    lines = [
        f"Pairwise-coincidence hit #{hit_index}   ({label_a} \u2194 {label_b})",
        "",
        "\u2500\u2500 Per-station \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"  {label_a:<8} freq {f_a:.6f} MHz   drift {d_a:+.5f} Hz/s   SNR {snr_a:.1f}",
        f"  {label_b:<8} freq {f_b:.6f} MHz   drift {d_b:+.5f} Hz/s   SNR {snr_b:.1f}",
        "",
        "\u2500\u2500 Delta \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"  {label_a} \u2013 {label_b}   \u0394freq {df_a_b:+.3f} Hz   \u0394drift {dd_a_b:+.5f} Hz/s",
        "",
        "\u2500\u2500 Tolerances applied \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500",
        f"  freq  \u00b1{freq_tol_hz:.0f} Hz",
        f"  drift \u00b1{drift_tol:.2f} Hz/s",
    ]

    ax.text(
        0.03, 0.90, "\n".join(lines),
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=9,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#aaaaaa'),
    )


# ── Main figure builder ────────────────────────────────────────────────────────

def make_pair_figure(
    row_a, row_b,
    hf_a, fch1_a, foff_a, tsamp_a, label_a,
    hf_b, fch1_b, foff_b, tsamp_b, label_b,
    half_width, hit_index,
    freq_tol_hz, drift_tol,
):
    """
    Build a figure (2 waterfalls on top, info panel below) for one
    pairwise-coincidence hit.
    Returns the Figure, or None if both stamps fail to extract.
    """
    fig = plt.figure(figsize=(12, 8))
    fig.suptitle(
        f"Pairwise-Coincidence Hit #{hit_index}  |  {label_a} \u2194 {label_b}  |  "
        f"~{row_a['_hit_freq_mhz']:.4f} MHz",
        fontsize=11, fontweight='bold', y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, height_ratios=[1.3, 1], hspace=0.4, wspace=0.32)
    ax_a   = fig.add_subplot(gs[0, 0])
    ax_b   = fig.add_subplot(gs[0, 1])
    ax_inf = fig.add_subplot(gs[1, :])

    station_cfg = [
        (label_a, row_a, hf_a, fch1_a, foff_a, tsamp_a, ax_a),
        (label_b, row_b, hf_b, fch1_b, foff_b, tsamp_b, ax_b),
    ]

    any_ok = False
    for label, row, hf, fch1, foff, tsamp, ax in station_cfg:
        hit_freq = float(row["_hit_freq_mhz"])
        try:
            stamp, freqs, times = make_stamp(hf, fch1, foff, tsamp, hit_freq, half_width)
            draw_waterfall(ax, stamp, freqs, times, label, row)
            any_ok = True
        except Exception as e:
            ax.axis('off')
            ax.text(0.5, 0.5, f"{label}\n(stamp error)\n{e}",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=8, color='red')

    draw_info(ax_inf, row_a, row_b, label_a, label_b, hit_index, freq_tol_hz, drift_tol)

    if not any_ok:
        plt.close(fig)
        return None
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot 3-panel pairwise-coincidence waterfall stamps (2 stations).'
    )
    # CSVs (matched rows output by anti_coincidence.py in 2-station mode)
    parser.add_argument('--csv_a', required=True, help='Matched hits CSV for station A')
    parser.add_argument('--csv_b', required=True, help='Matched hits CSV for station B')
    # H5 waterfall files
    parser.add_argument('--h5_a', required=True, help='.h5 waterfall file for station A')
    parser.add_argument('--h5_b', required=True, help='.h5 waterfall file for station B')
    # Labels for display / filenames
    parser.add_argument('--label_a', default='Station-A', help='Display label for station A (default: Station-A)')
    parser.add_argument('--label_b', default='Station-B', help='Display label for station B (default: Station-B)')
    # Options
    parser.add_argument('--width',      type=int,   default=524,
                        help='Total channel width of each stamp (default: 524)')
    parser.add_argument('--freq_tol',   type=float, default=10.0,
                        help='Freq tolerance in Hz used during matching (for display; default 10)')
    parser.add_argument('--drift_tol',  type=float, default=0.8,
                        help='Drift tolerance Hz/s used during matching (for display; default 0.8)')
    parser.add_argument('--outdir',     default='pair_stamps',
                        help='Output directory (default: pair_stamps/)')
    args = parser.parse_args()

    half_width = args.width // 2
    os.makedirs(args.outdir, exist_ok=True)

    # Load the two matched CSV files
    hits_a = load_matched_hits(args.csv_a)
    hits_b = load_matched_hits(args.csv_b)

    n = len(hits_a)
    
    if len(hits_b) != n:
        raise ValueError(
            f"Row counts don't match across the two CSVs: "
            f"{args.label_a}={len(hits_a)}, {args.label_b}={len(hits_b)}. "
            "Make sure you are passing the two *matched* output files from "
            "anti_coincidence.py's 2-station mode (e.g. hits_NA_SV_NA.csv / hits_NA_SV_SV.csv)."
        )
    if n == 0:
        print(f"No pairwise-coincidence hits found in the matched CSVs: {args.label_a} / {args.label_b}. Nothing to plot.")
        return

    print(f"Loaded {n} pairwise-coincidence hits ({args.label_a} \u2194 {args.label_b}).")

    # Open both H5 files
    hf_a, fch1_a, foff_a, tsamp_a = open_h5(args.h5_a)
    hf_b, fch1_b, foff_b, tsamp_b = open_h5(args.h5_b)

    pdf_path = os.path.join(args.outdir, 'pair_coincidence_stamps.pdf')

    try:
        with mpdf.PdfPages(pdf_path) as pdf:
            for i in range(n):
                row_a = hits_a.iloc[i]
                row_b = hits_b.iloc[i]
                hit_index = i + 1

                fig = make_pair_figure(
                    row_a, row_b,
                    hf_a, fch1_a, foff_a, tsamp_a, args.label_a,
                    hf_b, fch1_b, foff_b, tsamp_b, args.label_b,
                    half_width, hit_index,
                    args.freq_tol, args.drift_tol,
                )

                if fig is None:
                    print(f"  Hit #{hit_index}: both stamps failed, skipping page.")
                    continue

                freq_label = f"{row_a['_hit_freq_mhz']:.4f}MHz"
                png_name   = f"pair_hit_{hit_index:04d}_{freq_label}.png"
                png_path   = os.path.join(args.outdir, png_name)
                fig.savefig(png_path, dpi=150, bbox_inches='tight')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                print(
                    f"  [{hit_index}/{n}]  {freq_label}  "
                    f"SNR {args.label_a}={row_a['SNR']:.1f} {args.label_b}={row_b['SNR']:.1f}  "
                    f"-> {png_name}"
                )
    finally:
        hf_a.close()
        hf_b.close()

    print(f"\nDone. PNGs + PDF in '{args.outdir}/'")
    print(f"PDF: {pdf_path}")


if __name__ == '__main__':
    main()

# Example usage:
""""
python plot_pair_coincidence.py
--csv_a results/hits_ONE_SV_ONE.csv --h5_a frb_set_two/059613_002567249-LWA1_tun2.h5 --label_a LWA1
--csv_b results/hits_ONE_SV_SV.csv --h5_b frb_set_two/059613_002172869-LWA-SV_tun2.h5 --label_b LWA-SV
--width 524 --freq_tol 100 --drift_tol 10.0 --outdir pair_stamps_frb_set_two_tun2
"""

# python3 second_pipeline_scripts/plot_pair_coincidence.py --csv_a frb_data/frb_8/results_frb_8/tun2/hits_ONE_SV_SV.csv --h5_a frb_data/frb_8/059639_002431255-LWA-SV_tun2.h5 --label_a LWASV --csv_b frb_data/frb_8/results_frb_8/tun2/hits_ONE_SV_ONE.csv --h5_b frb_data/frb_8/059639_002868043-LWA1_tun2.h5 --label_b LWA1 --width 524 --freq_tol 100 --drift_tol 10.0 --outdir pair_stamps_frb_8_compare
