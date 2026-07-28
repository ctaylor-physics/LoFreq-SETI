"""
lwa_bliss_bp_gen.py

Stage 1 of the LWA coincidence-search pipeline (see run_pipeline.py).

Given a single LWA waterfall .h5 file, this script:
    1. Computes an *instrumental* bandpass model from the station's known
       hardware response (antenna impedance mismatch, ARX filter response,
       and DRX filter response) — this captures the shape imposed by the
       receiver hardware itself, independent of the data.
    2. Divides that instrumental shape out of a handful of sample rows from
       the data, then fits a smooth (Savitzky-Golay) curve to what's left.
       That residual is the leftover "PFB ripple" — a periodic ripple
       pattern introduced by the polyphase filter bank used to channelize
       the data. Combines it with the instrumental model to get one
       complete bandpass model, which is saved into the .h5 file (as
       'bandpass_model') and used to write a bandpass-corrected copy of the
       data (as 'corrected') back into the same file.
    3. Smooths that combined bandpass model once more and writes it out as
       a standalone flat binary float32 file: '<basename>_bpmodel.f32'.
       This .f32 file is the only thing run_pipeline.py consumes from this
       script — it gets passed as --bp to chunk_and_bliss.py downstream.

Usage
-----
    python lwa_bliss_bp_gen.py path/to/data.h5

Produces (in the current working directory):
    <basename>_bpmodel.f32
"""

import sys
import os
import numpy as np
import h5py

from lsl.common import stations, ndp
from lsl.reader.drx import FILTER_CODES

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# CuPy is optional — if it's installed, the chunked correction pass in
# computeBandpassData_streaming() runs on GPU. If not, we fall back to
# plain NumPy (slower, but functionally identical).
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy found — GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found — falling back to CPU")


# ── instrumental bandpass (hardware-derived, not fit from data) ────────────

def getImpedanceMisMatch(freq):
    """
    Interpolate the antenna's impedance-mismatch response (from LSL's
    station model) onto an arbitrary set of frequencies.
    """
    freq4, imm4 = stations.lwa1.antennas[0].response(dB=False)
    immIntp = interp1d(freq4, imm4, kind='cubic', bounds_error=False)
    return immIntp(freq)


def getARXResponse(freq, filter='full', site=stations.lwa1):
    """
    Average the ARX (analog receiver) response across all "good" antennas
    (status code 33), separately for the X and Y polarizations, then
    interpolate onto the requested frequency grid.
    """
    antennas = site.antennas
    f, r = antennas[0].arx.response(filter='split')
    freq2 = f
    respX2 = np.zeros_like(r)
    respY2 = np.zeros_like(r)

    for i in range(len(antennas)):
        if antennas[i].combined_status != 33:
            continue
        f, r = antennas[i].arx.response(filter=filter, dB=False)
        if antennas[i].pol == 0:
            respX2 += r
        else:
            respY2 += r

    respX2 /= respX2.max()
    respY2 /= respY2.max()

    respXIntp = interp1d(freq2, respX2, kind='cubic', bounds_error=False)
    respYIntp = interp1d(freq2, respY2, kind='cubic', bounds_error=False)

    respX = respXIntp(freq)
    respY = respYIntp(freq)
    # interp1d returns NaN outside its domain — zero those out instead
    respX = np.where(np.isfinite(respX), respX, 0)
    respY = np.where(np.isfinite(respY), respY, 0)
    return respX, respY


def getDRXResponse(freq, filterCode=7):
    """
    DRX (digital receiver) filter response for the given filter code,
    evaluated relative to the band center.
    """
    srate = FILTER_CODES[filterCode]
    dpf = ndp.drx_filter(sample_rate=srate)
    return dpf(freq - freq.mean())


def compute_bandpass_instrumental(freq):
    """
    Combine impedance mismatch, ARX, and DRX responses into one normalized
    instrumental bandpass model. `freq` must be in Hz.
    """
    rIMM = getImpedanceMisMatch(freq)
    rDRX = getDRXResponse(freq, filterCode=7)
    rARXX, rARXY = getARXResponse(freq, filter='split')
    rARX = 0.5 * rARXX + 0.5 * rARXY

    bpm = rIMM / rIMM.max() * rARX / rARX.max() * rDRX / rDRX.max()
    return bpm


# ── smoothing helper ────────────────────────────────────────────────────────

def smooth_bpmodel(bpmodel, od=9, ws_min=41):
    """
    Apply a Savitzky-Golay smoothing pass to a 1-D bandpass model and
    renormalize it to a mean of 1. `ws_min` is accepted for API
    compatibility but the window width is actually derived from
    sqrt(len(bpmodel)) below (matching the original behavior).
    """
    ws = int(round(np.sqrt(len(bpmodel))))
    if ws % 2 == 0:
        ws += 1

    bpm = savgol_filter(bpmodel, ws, od, deriv=0)
    bpm = np.ma.array(bpm, mask=~np.isfinite(bpm))

    if bpm.mean() == 0:
        bpm += 1
    bpm = bpm / bpm.mean()
    return bpm


# ── residual PFB-ripple fit + in-place HDF5 correction ─────────────────────

def computeBandpassData_streaming(
    hdf5_path,
    input_dataset,
    output_dataset="corrected",
    bpm_dataset="bandpass_model",
    chunk_size=16,
    bpm_estimation_chunks=32,
    instr_bandpass=None,
    window_size=41,
):
    """
    Streaming PFB-ripple correction for a large HDF5 spectral dataset.

    Pass 1: sample a handful of rows spread evenly across the file, divide
    out the instrumental bandpass (if given), take the median across those
    rows, and fit a Savitzky-Golay curve to what's left. That's the
    "residual ripple" — the PFB channelizer's periodic gain wobble that
    the instrumental model doesn't already account for.

    The residual ripple is combined with the instrumental model into one
    normalized `combined_bpm`, which is written to the file as a 1-D
    dataset (`bpm_dataset`).

    Pass 2: stream through the file in chunks, divide every row by
    `combined_bpm`, and write the corrected data to `output_dataset`.

    Returns
    -------
    combined_bpm : np.ndarray, shape [F]
        The full normalized bandpass model applied to the data.
    """
    xp = cp if GPU_AVAILABLE else np

    with h5py.File(hdf5_path, "a") as f:
        ds = f[input_dataset]
        raw_shape = ds.shape
        dtype = ds.dtype

        if ds.ndim == 3:
            N, _, F = raw_shape
            squeeze = True
        elif ds.ndim == 2:
            N, F = raw_shape
            squeeze = False
        else:
            raise ValueError(f"Expected 2-D or 3-D dataset, got shape {raw_shape}")

        print(f"Dataset  : {input_dataset}  {raw_shape}  ({dtype})")
        print(f"Rows N={N}, Channels F={F}")
        print(f"Backend  : {'GPU (CuPy)' if GPU_AVAILABLE else 'CPU (NumPy)'}")

        # Instrumental model must match the data's channel count. Normalize
        # to a mean of 1 so it only reshapes the spectrum, doesn't rescale it.
        if instr_bandpass is not None:
            instr_bandpass = np.asarray(instr_bandpass, dtype=np.float64)
            if instr_bandpass.shape != (F,):
                raise ValueError(
                    f"instr_bandpass shape {instr_bandpass.shape} does not "
                    f"match channel axis F={F}"
                )
            instr_bandpass = instr_bandpass / instr_bandpass.mean()
            print(f"Instr. bandpass provided  |  "
                  f"min={instr_bandpass.min():.4f}  max={instr_bandpass.max():.4f}")
        else:
            print("Instr. bandpass          : None (fitting raw spectrum)")

        # ── PASS 1: estimate residual PFB ripple from a handful of rows ──
        sample_idx = np.linspace(0, N - 1, min(bpm_estimation_chunks, N), dtype=int)
        row_medians = np.empty((len(sample_idx), F), dtype=np.float64)

        print(f"\nPass 1 – estimating bpm from {len(sample_idx)} sampled rows …")
        for out_i, row_i in enumerate(sample_idx):
            raw = ds[row_i]
            row = raw[0].astype(np.float64) if squeeze else raw.astype(np.float64)

            if instr_bandpass is not None:
                row = row / instr_bandpass

            row_medians[out_i] = row
            print(f"  sampling row {out_i+1}/{len(sample_idx)}  (file row {row_i})", end="\r")

        meanSpec = xp.nanmedian(xp.array(row_medians), axis=0)
        meanSpec_cpu = cp.asnumpy(meanSpec) if GPU_AVAILABLE else meanSpec
        print(meanSpec_cpu.shape)

        # Window/order for the Savitzky-Golay fit, capped at window_size
        ws = int(round(F / 10.0))
        ws = min(window_size, ws)
        if ws % 2 == 0:
            ws += 1
        od = min(9, ws - 2)

        residual_ripple = savgol_filter(meanSpec_cpu, ws, od, deriv=0)
        if np.all(np.isnan(residual_ripple)):
            raise ValueError("savgol_filter returned all NaNs — meanSpec is likely all NaN or zero")

        residual_ripple = np.ma.array(residual_ripple, mask=~np.isfinite(residual_ripple))
        if residual_ripple.mean() == 0:
            residual_ripple += 1
        residual_ripple = np.asarray(residual_ripple / residual_ripple.mean(), dtype=np.float64)

        print(f"\nResidual ripple fit  (ws={ws}, od={od})  |  "
              f"min={residual_ripple.min():.4f}  max={residual_ripple.max():.4f}")

        # ── combine instrumental model x residual ripple = full bandpass ─
        if instr_bandpass is not None:
            combined_bpm = instr_bandpass * residual_ripple
            combined_bpm = combined_bpm / combined_bpm.mean()
            print(f"Combined bpm (instr × ripple)  |  "
                  f"min={combined_bpm.min():.4f}  max={combined_bpm.max():.4f}")
        else:
            combined_bpm = residual_ripple
            print("Combined bpm = residual ripple (no instr. bandpass supplied)")

        bpm_gpu = xp.array(combined_bpm)  # pushed once, reused every chunk below

        # ── write the 1-D combined bandpass model dataset ─────────────────
        if bpm_dataset in f:
            del f[bpm_dataset]
        bpm_ds = f.create_dataset(
            bpm_dataset, data=combined_bpm, dtype=np.float64,
            compression="gzip", compression_opts=4,
        )
        bpm_ds.attrs["savgol_window"] = ws
        bpm_ds.attrs["savgol_order"] = od
        bpm_ds.attrs["estimation_rows_used"] = len(sample_idx)
        bpm_ds.attrs["source_dataset"] = input_dataset
        bpm_ds.attrs["instrumental_bandpass"] = instr_bandpass is not None
        bpm_ds.attrs["components"] = (
            "instrumental x residual_pfb_ripple"
            if instr_bandpass is not None else "residual_pfb_ripple_only"
        )
        print(f"Saved combined bpm → '{bpm_dataset}'  shape {combined_bpm.shape}  float64")

        # ── create the corrected output dataset ───────────────────────────
        if output_dataset in f:
            del f[output_dataset]
        out_ds = f.create_dataset(
            output_dataset, shape=raw_shape, dtype=np.float32,
            chunks=(1, 1, min(F, 131072)) if squeeze else (1, min(F, 131072)),
            compression="gzip", compression_opts=4,
        )
        out_ds.attrs["bandpass_correction"] = "savgol_median"
        out_ds.attrs["savgol_window"] = ws
        out_ds.attrs["savgol_order"] = od
        out_ds.attrs["source_dataset"] = input_dataset
        out_ds.attrs["bpm_dataset"] = bpm_dataset
        out_ds.attrs["instrumental_bandpass"] = instr_bandpass is not None
        print(f"Created output dataset '{output_dataset}'  {raw_shape}  float32")

        # ── PASS 2: stream through the file, divide by combined_bpm ──────
        print(f"\nPass 2 – correcting in chunks of {chunk_size} rows …")
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = ds[start:end]
            data = chunk[:, 0, :].astype(np.float64) if squeeze else chunk.astype(np.float64)

            corrected = (xp.array(data) / bpm_gpu).astype(xp.float32)
            corrected_cpu = cp.asnumpy(corrected) if GPU_AVAILABLE else corrected

            if squeeze:
                out_ds[start:end, 0, :] = corrected_cpu
            else:
                out_ds[start:end, :] = corrected_cpu

            pct = 100.0 * end / N
            print(f"  rows {start:>4}–{end:>4} / {N}  ({pct:.1f}%)", end="\r")

        print(f"\nDone. '{output_dataset}' and '{bpm_dataset}' written to {hdf5_path}")

    return combined_bpm


# ── main ─────────────────────────────────────────────────────────────────────

def main(TESTDATA, OUTFILE):
    with h5py.File(TESTDATA, 'r') as f:
        # Frequency axis is stored as start freq (fch1) + per-channel offset (foff), in MHz
        fch1 = f['data'].attrs['fch1']
        foff = f['data'].attrs['foff']
        nchans = f['data'].attrs['nchans']
        freqs = fch1 + foff * np.arange(nchans)
        keys = list(f.keys())

    # 1. Instrumental bandpass, normalized to [0, 1] with a tiny floor to
    #    avoid divide-by-zero later.
    bpm_i = compute_bandpass_instrumental(freqs * 1e6)  # needs Hz, freqs is MHz
    bpm_i = (bpm_i - bpm_i.min()) / (bpm_i.max() - bpm_i.min())
    bpm_i[bpm_i == 0.0] += 1e-16

    # 2. If this file hasn't already been bandpass-corrected, fit the
    #    residual PFB ripple and write 'corrected' + 'bandpass_model' into
    #    the HDF5 file. Skip if it's already been done (e.g. re-running
    #    the pipeline on the same file).
    if 'corrected' not in keys:
        print('Uncorrected File, making the fit!')
        computeBandpassData_streaming(
            hdf5_path=TESTDATA,
            input_dataset="data",
            output_dataset="corrected",
            bpm_dataset="bandpass_model",
            bpm_estimation_chunks=32,
            instr_bandpass=bpm_i,
            window_size=41,
        )
    else:
        print('Corrected File, proceed!')

    # 3. Read back the combined bandpass model, smooth it once more, and
    #    write it out as the standalone .f32 file that run_pipeline.py
    #    passes to chunk_and_bliss.py as --bp.
    with h5py.File(TESTDATA, 'r') as f:
        bpm_m = f['bandpass_model'][:]

    bpm5 = smooth_bpmodel(bpm_m, od=4)
    bpm5 = bpm5.data.astype(np.float32)
    bpm5.tofile(OUTFILE)
    return


if __name__ == "__main__":
    filename = sys.argv[-1]
    outfile = os.path.basename(filename).split('.')[0] + "_bpmodel.f32"
    main(filename, outfile)
