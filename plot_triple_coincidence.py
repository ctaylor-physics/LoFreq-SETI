"""
plot_triple_coincidence.py

For each triple-coincidence hit (common across LWA-NA, LWA-SV, LWA-1),
produce a 4-quadrant figure:

    ┌──────────┬──────────┐
    │  LWA-NA  │  LWA-SV  │
    ├──────────┼──────────┤
    │  LWA-1   │  INFO    │
    └──────────┴──────────┘

The INFO quadrant summarises the hit: frequencies, drift rates, SNRs,
and the pairwise Δfreq / Δdrift between stations.

Inputs
------
Three matched CSV files produced by anti_coincidence.py
(hits_all_three_NA.csv, hits_all_three_SV.csv, hits_all_three_ONE.csv)
and the corresponding .h5 waterfall file for each station.

Usage
-----
python3 plot_triple_coincidence.py \\
    --na_csv  hits_all_three_NA.csv  --na_h5  lwa_na.h5  \\
    --sv_csv  hits_all_three_SV.csv  --sv_h5  lwa_sv.h5  \\
    --one_csv hits_all_three_ONE.csv --one_h5 lwa_one.h5 \\
    --width 524 --outdir triple_stamps
"""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf
import matplotlib.gridspec as gridspec
import numpy as np

# ── Reused helpers (identical logic to plot_bliss_stamps.py) ──────────────────
from plot_bliss_stamps import load_hits, make_stamp

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
    ax.set_xlabel('Frequency (MHz)', fontsize=8)
    ax.set_ylabel('Time (s)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(
        f"{label}  |  {hit_row['_hit_freq_mhz']:.6f} MHz  "
        f"|  SNR {hit_row['SNR']:.1f}  |  drift {hit_row['Drift_Rate']:.5f} Hz/s",
        fontsize=7.5,
    )
    plt.colorbar(im, ax=ax, label='Power', pad=0.02)


# ── Info quadrant ──────────────────────────────────────────────────────────────

def draw_info(ax, row_na, row_sv, row_one, hit_index, freq_tol_hz, drift_tol):
    """Fill the bottom-right quadrant with hit metadata and pairwise deltas."""
    ax.axis('off')

    f_na  = row_na['_hit_freq_mhz'];  d_na  = row_na['Drift_Rate'];  snr_na  = row_na['SNR']
    f_sv  = row_sv['_hit_freq_mhz'];  d_sv  = row_sv['Drift_Rate'];  snr_sv  = row_sv['SNR']
    f_one = row_one['_hit_freq_mhz']; d_one = row_one['Drift_Rate']; snr_one = row_one['SNR']

    df_na_sv   = (f_na  - f_sv)  * 1e6   # MHz → Hz
    df_na_one  = (f_na  - f_one) * 1e6
    df_sv_one  = (f_sv  - f_one) * 1e6
    dd_na_sv   = d_na  - d_sv
    dd_na_one  = d_na  - d_one
    dd_sv_one  = d_sv  - d_one

    lines = [
        f"Triple-coincidence hit #{hit_index}",
        "",
        "── Per-station ──────────────────",
        f"  LWA-NA   freq {f_na:.6f} MHz   drift {d_na:+.5f} Hz/s   SNR {snr_na:.1f}",
        f"  LWA-SV   freq {f_sv:.6f} MHz   drift {d_sv:+.5f} Hz/s   SNR {snr_sv:.1f}",
        f"  LWA-1    freq {f_one:.6f} MHz   drift {d_one:+.5f} Hz/s   SNR {snr_one:.1f}",
        "",
        "── Pairwise Δ ───────────────────",
        f"  NA – SV   Δfreq {df_na_sv:+.3f} Hz   Δdrift {dd_na_sv:+.5f} Hz/s",
        f"  NA – 1    Δfreq {df_na_one:+.3f} Hz   Δdrift {dd_na_one:+.5f} Hz/s",
        f"  SV – 1    Δfreq {df_sv_one:+.3f} Hz   Δdrift {dd_sv_one:+.5f} Hz/s",
        "",
        "── Tolerances applied ───────────",
        f"  freq  ±{freq_tol_hz:.0f} Hz",
        f"  drift ±{drift_tol:.2f} Hz/s",
    ]

    ax.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax.transAxes,
        va='top', ha='left',
        fontsize=8,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor='#aaaaaa'),
    )


# ── Main figure builder ────────────────────────────────────────────────────────

def make_triple_figure(
    row_na, row_sv, row_one,
    hf_na, fch1_na, foff_na, tsamp_na,
    hf_sv, fch1_sv, foff_sv, tsamp_sv,
    hf_one, fch1_one, foff_one, tsamp_one,
    half_width, hit_index,
    freq_tol_hz, drift_tol,
):
    """
    Build a 2×2 figure for one triple-coincidence hit.
    Returns the Figure, or None if all three stamps fail to extract.
    """
    fig = plt.figure(figsize=(14, 9))
    fig.suptitle(
        f"Triple-Coincidence Hit #{hit_index}  |  "
        f"~{row_na['_hit_freq_mhz']:.4f} MHz",
        fontsize=11, fontweight='bold', y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    ax_na  = fig.add_subplot(gs[0, 0])
    ax_sv  = fig.add_subplot(gs[0, 1])
    ax_one = fig.add_subplot(gs[1, 0])
    ax_inf = fig.add_subplot(gs[1, 1])

    station_cfg = [
        ("LWA-NA",  row_na,  hf_na,  fch1_na,  foff_na,  tsamp_na,  ax_na),
        ("LWA-SV",  row_sv,  hf_sv,  fch1_sv,  foff_sv,  tsamp_sv,  ax_sv),
        ("LWA-1",   row_one, hf_one, fch1_one, foff_one, tsamp_one, ax_one),
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

    draw_info(ax_inf, row_na, row_sv, row_one, hit_index, freq_tol_hz, drift_tol)

    if not any_ok:
        plt.close(fig)
        return None
    return fig


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Plot 4-quadrant triple-coincidence waterfall stamps.'
    )
    # CSVs (matched rows output by anti_coincidence.py)
    parser.add_argument('--na_csv',  required=True, help='Triple-matched hits CSV for LWA-NA')
    parser.add_argument('--sv_csv',  required=True, help='Triple-matched hits CSV for LWA-SV')
    parser.add_argument('--one_csv', required=True, help='Triple-matched hits CSV for LWA-1')
    # H5 waterfall files
    parser.add_argument('--na_h5',  required=True, help='.h5 waterfall file for LWA-NA')
    parser.add_argument('--sv_h5',  required=True, help='.h5 waterfall file for LWA-SV')
    parser.add_argument('--one_h5', required=True, help='.h5 waterfall file for LWA-1')
    # Options
    parser.add_argument('--width',      type=int,   default=524,
                        help='Total channel width of each stamp (default: 524)')
    parser.add_argument('--freq_tol',   type=float, default=10.0,
                        help='Freq tolerance in Hz used during matching (for display; default 10)')
    parser.add_argument('--drift_tol',  type=float, default=0.8,
                        help='Drift tolerance Hz/s used during matching (for display; default 0.8)')
    parser.add_argument('--outdir',     default='triple_stamps',
                        help='Output directory (default: triple_stamps/)')
    args = parser.parse_args()

    half_width = args.width // 2
    os.makedirs(args.outdir, exist_ok=True)

    # Load the three matched CSV files
    hits_na  = load_hits(args.na_csv)
    hits_sv  = load_hits(args.sv_csv)
    hits_one = load_hits(args.one_csv)

    n = len(hits_na)
    if not (len(hits_sv) == n == len(hits_one)):
        raise ValueError(
            f"Row counts don't match across the three CSVs: "
            f"NA={len(hits_na)}, SV={len(hits_sv)}, ONE={len(hits_one)}. "
            "Make sure you are passing the three *matched* output files from "
            "anti_coincidence.py (hits_all_three_NA/SV/ONE.csv)."
        )

    print(f"Loaded {n} triple-coincidence hits.")

    # Open all three H5 files
    hf_na,  fch1_na,  foff_na,  tsamp_na  = open_h5(args.na_h5)
    hf_sv,  fch1_sv,  foff_sv,  tsamp_sv  = open_h5(args.sv_h5)
    hf_one, fch1_one, foff_one, tsamp_one = open_h5(args.one_h5)

    pdf_path = os.path.join(args.outdir, 'triple_coincidence_stamps.pdf')

    try:
        with mpdf.PdfPages(pdf_path) as pdf:
            for i in range(n):
                row_na  = hits_na.iloc[i]
                row_sv  = hits_sv.iloc[i]
                row_one = hits_one.iloc[i]
                hit_index = i + 1

                fig = make_triple_figure(
                    row_na, row_sv, row_one,
                    hf_na,  fch1_na,  foff_na,  tsamp_na,
                    hf_sv,  fch1_sv,  foff_sv,  tsamp_sv,
                    hf_one, fch1_one, foff_one, tsamp_one,
                    half_width, hit_index,
                    args.freq_tol, args.drift_tol,
                )

                if fig is None:
                    print(f"  Hit #{hit_index}: all stamps failed, skipping page.")
                    continue

                freq_label = f"{row_na['_hit_freq_mhz']:.4f}MHz"
                png_name   = f"triple_hit_{hit_index:04d}_{freq_label}.png"
                png_path   = os.path.join(args.outdir, png_name)
                fig.savefig(png_path, dpi=150, bbox_inches='tight')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

                print(
                    f"  [{hit_index}/{n}]  {freq_label}  "
                    f"SNR NA={row_na['SNR']:.1f} SV={row_sv['SNR']:.1f} "
                    f"ONE={row_one['SNR']:.1f}  -> {png_name}"
                )
    finally:
        hf_na.close()
        hf_sv.close()
        hf_one.close()

    print(f"\nDone. PNGs + PDF in '{args.outdir}/'")
    print(f"PDF: {pdf_path}")


if __name__ == '__main__':
    main()

# Example usage:
# python3 plot_triple_coincidence.py \
#     --na_csv  results_day_obs/tun2/hits_all_three_NA.csv  --na_h5  day_obs_tunings/061196_008848197-LWA-NA_tun2.h5  \
#     --sv_csv  results_day_obs/tun2/hits_all_three_SV.csv  --sv_h5  day_obs_tunings/061196_000885206-LWA-SV_tun2.h5  \
#     --one_csv results_day_obs/tun2/hits_all_three_ONE.csv --one_h5 day_obs_tunings/061196_001478830-LWA1_tun2.h5 \
#     --width 524 --freq_tol 10 --drift_tol 0.8 --outdir triple_stamps

# python3 plot_triple_coincidence.py --na_csv  anti_aligned_test/hits_all_three_NA.csv  --na_h5  day_obs_tunings/061196_008848197-LWA-NA_tun2.h5 --sv_csv  anti_aligned_test/hits_all_three_SV.csv  --sv_h5  day_obs_tunings/061196_000885206-LWA-SV_tun2.h5 --one_csv anti_aligned_test/hits_all_three_ONE.csv --one_h5 day_obs_tunings/061196_001478830-LWA1_tun2.h5 --width 524 --freq_tol 10 --drift_tol 0.8 --outdir triple_stamps_2