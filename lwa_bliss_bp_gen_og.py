"""
** Original code from Craig Taylor, used for creating a bandpass model and signal injection testing **

This code is to test what the recovery looks like 
when injecting SETI signals into LWA Data
1. load in a dataset from the LWA 
2. correct the bandpass to get a normalized spectrum
3. inject array of signals
4. Write the bliss call that we used
5. Plot the results against the injected signals
"""
#!/usr/bin/env python3

import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import os

from lsl.common import stations, ndp
from lsl.reader.drx import FILTER_CODES

from scipy.stats import scoreatpercentile as percentile, skew, kurtosis
from scipy.interpolate import interp1d
from scipy.signal import get_window, firwin, correlate, find_peaks, medfilt, savgol_filter
from scipy.optimize import least_squares


try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("CuPy found — GPU acceleration enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("CuPy not found — falling back to CPU")

def getImpedanceMisMatch(freq):
    freq4, imm4 = stations.lwa1.antennas[0].response(dB=False)

    immIntp = interp1d(freq4, imm4, kind='cubic', bounds_error=False)

    imm = immIntp(freq)
    return imm

def getARXResponse(freq, filter='full', site=stations.lwa1):
    antennas = site.antennas
    f,r = antennas[0].arx.response(filter='split')
    freq2 = f
    respX2 = np.zeros_like(r)
    respY2 = np.zeros_like(r)
    for i in range(len(antennas)):
        if antennas[i].combined_status != 33:
            continue
        f,r = antennas[i].arx.response(filter=filter, dB=False)

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
    respX = np.where(np.isfinite(respX), respX, 0)
    respY = np.where(np.isfinite(respY), respY, 0)
    return respX, respY

def getDRXResponse(freq, filterCode=7):
    srate = FILTER_CODES[filterCode]
    dpf = ndp.drx_filter(sample_rate=srate)

    rDRX = dpf(freq-freq.mean())

    return rDRX

def compute_bandpass_instrumental(freq): # Must be in Hz
    
    # freq is all frequencies [Hz]
    rIMM = getImpedanceMisMatch(freq)

    ## DRX Response
    rDRX = getDRXResponse(freq, filterCode=7)

    ## ARX Response
    rARXX, rARXY= getARXResponse(freq, filter='split')
    rARX = 0.5*rARXX + 0.5*rARXY
    # rARX = 1.0*rARXX

    ## Compute bpm
    bpm = rIMM/rIMM.max() * rARX/rARX.max() * rDRX/rDRX.max()
    return bpm

def pfb_window(P): #10700 for 8388608 seti data
    win_coeffs = get_window("hamming", 4*P)
    sinc       = firwin(4*P, cutoff=1.0/P, window="rectangular")
    win_coeffs *= sinc
    win_coeffs /= win_coeffs.max()
    return win_coeffs

def smooth_bpmodel(bpmodel, od=9, ws_min=41):

    # Come up with an appropriate smoothing window (wd) and order (od)
    ws = int(round(np.sqrt(len(bpmodel))))
    
    # ws = min([41, ws])
    # ws = min([ws_min, ws])

    if ws % 2 == 0:
        ws += 1
    # od = min([9, ws-2])

    bpm = savgol_filter(bpmodel, ws, od, deriv=0)
    bpm = np.ma.array(bpm, mask=~np.isfinite(bpm))

    if bpm.mean() == 0:
        bpm += 1
    bpm = bpm / bpm.mean()

    return bpm

def computeBandpassData(spec, ws_min=41):
    """
    Compute data-based bandpass fits.
    Adapted from lwa-project -> commissioning -> plotHDF
    """

    meanSpec = np.nanmedian(spec, axis=0, overwrite_input=True)

    # Come up with an appropriate smoothing window (wd) and order (od)
    ws = int(round(spec.shape[1]/10.0))
    # ws = min([41, ws])
    ws = min([ws_min, ws])
    if ws % 2 == 0:
        ws += 1
    od = min([9, ws-2])

    bpm = savgol_filter(meanSpec, ws, od, deriv=0)
    bpm = np.ma.array(bpm, mask=~np.isfinite(bpm))

    if bpm.mean() == 0:
        bpm += 1
    bpm = bpm / bpm.mean()

    # Apply the bandpass correction
    specBandpass = spec / bpm

    return specBandpass, bpm

def computeBandpassData_streaming(
    hdf5_path,
    input_dataset,
    output_dataset="corrected",
    bpm_dataset="bandpass_model",
    chunk_size=16,
    bpm_estimation_chunks=32,
    instr_bandpass=None,
    window_size = 41
):
    """
    Streaming PFB ripple correction for large HDF5 spectral datasets.

    Optionally accepts an instrumental bandpass model (instr_bandpass) which
    is divided out before fitting for the residual PFB ripple. The final
    combined bandpass model (instrumental × residual ripple) is saved as a
    1-D dataset and applied to the data.

    Parameters
    ----------
    hdf5_path : str
        Path to the HDF5 file.
    input_dataset : str
        Key of the source dataset, shape [N, 1, F] or [N, F].
    output_dataset : str
        Key under which the corrected dataset will be written.
    bpm_dataset : str
        Key under which the final combined 1-D bandpass model will be written.
    chunk_size : int
        Number of rows to read/write per iteration.
    bpm_estimation_chunks : int
        Number of rows (spread evenly) used to estimate the median spectrum.
    instr_bandpass : np.ndarray or None, shape [F]
        Optional instrumental bandpass model. If provided, this is divided
        out before fitting the residual PFB ripple. The final saved model
        is the product of instr_bandpass and the fitted residual ripple,
        giving a fully described bandpass.

    Returns
    -------
    combined_bpm : np.ndarray, shape [F]
        The full normalised bandpass model applied to the data.
        If instr_bandpass=None this is just the fitted PFB ripple.
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

        # ── Validate instr_bandpass ───────────────────────────────────────
        if instr_bandpass is not None:
            instr_bandpass = np.asarray(instr_bandpass, dtype=np.float64)
            if instr_bandpass.shape != (F,):
                raise ValueError(
                    f"instr_bandpass shape {instr_bandpass.shape} does not "
                    f"match channel axis F={F}"
                )
            # Normalise so it doesn't rescale the data, only removes its shape
            instr_bandpass = instr_bandpass / instr_bandpass.mean()
            print(f"Instr. bandpass provided  |  "
                  f"min={instr_bandpass.min():.4f}  max={instr_bandpass.max():.4f}")
        else:
            print("Instr. bandpass          : None (fitting raw spectrum)")

        # ── PASS 1 : estimate residual PFB ripple ─────────────────────────
        sample_idx = np.linspace(0, N - 1, min(bpm_estimation_chunks, N), dtype=int)
        row_medians = np.empty((len(sample_idx), F), dtype=np.float64)

        print(f"\nPass 1 – estimating bpm from {len(sample_idx)} sampled rows …")
        for out_i, row_i in enumerate(sample_idx):
            raw = ds[row_i]
            row = raw[0].astype(np.float64) if squeeze else raw.astype(np.float64)

            # Divide out instrumental shape before accumulating
            if instr_bandpass is not None:
                row = row / instr_bandpass

            row_medians[out_i] = row
            print(f"  sampling row {out_i+1}/{len(sample_idx)}  (file row {row_i})", end="\r")

        # Median and Savitzky-Golay fit to residual ripple
        meanSpec = xp.nanmedian(xp.array(row_medians), axis=0)
        meanSpec_cpu = cp.asnumpy(meanSpec) if GPU_AVAILABLE else meanSpec

        print(meanSpec_cpu.shape)

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
        residual_ripple = np.asarray(residual_ripple / residual_ripple.mean(),
                                     dtype=np.float64)

        print(f"\nResidual ripple fit  (ws={ws}, od={od})  |  "
              f"min={residual_ripple.min():.4f}  max={residual_ripple.max():.4f}")

        # ── Combine into full bandpass model ──────────────────────────────
        if instr_bandpass is not None:
            combined_bpm = instr_bandpass * residual_ripple
            combined_bpm = combined_bpm / combined_bpm.mean()  # renormalise product
            print(f"Combined bpm (instr × ripple)  |  "
                  f"min={combined_bpm.min():.4f}  max={combined_bpm.max():.4f}")
        else:
            combined_bpm = residual_ripple
            print("Combined bpm = residual ripple (no instr. bandpass supplied)")

        # Push combined model to GPU once — reused every chunk
        bpm_gpu = xp.array(combined_bpm)

        # ── Write 1-D bandpass model dataset ──────────────────────────────
        if bpm_dataset in f:
            del f[bpm_dataset]
        bpm_ds = f.create_dataset(
            bpm_dataset, data=combined_bpm, dtype=np.float64,
            compression="gzip", compression_opts=4,
        )
        bpm_ds.attrs["savgol_window"]            = ws
        bpm_ds.attrs["savgol_order"]             = od
        bpm_ds.attrs["estimation_rows_used"]     = len(sample_idx)
        bpm_ds.attrs["source_dataset"]           = input_dataset
        bpm_ds.attrs["instrumental_bandpass"]    = instr_bandpass is not None
        bpm_ds.attrs["components"]               = (
            "instrumental x residual_pfb_ripple"
            if instr_bandpass is not None else "residual_pfb_ripple_only"
        )
        print(f"Saved combined bpm → '{bpm_dataset}'  shape {combined_bpm.shape}  float64")

        # ── Create output dataset ─────────────────────────────────────────
        if output_dataset in f:
            del f[output_dataset]
        out_ds = f.create_dataset(
            output_dataset, shape=raw_shape, dtype=np.float32,
            chunks=(1, 1, min(F, 131072)) if squeeze else (1, min(F, 131072)),
            compression="gzip", compression_opts=4,
        )
        out_ds.attrs["bandpass_correction"]      = "savgol_median"
        out_ds.attrs["savgol_window"]            = ws
        out_ds.attrs["savgol_order"]             = od
        out_ds.attrs["source_dataset"]           = input_dataset
        out_ds.attrs["bpm_dataset"]              = bpm_dataset
        out_ds.attrs["instrumental_bandpass"]    = instr_bandpass is not None
        print(f"Created output dataset '{output_dataset}'  {raw_shape}  float32")

        # ── PASS 2 : stream, correct with combined bpm, write ─────────────
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

def bin_spectrum(x, y, bin_size=1024):
    """
    Median-bin a long 1D spectrum.
    """
    n = len(y)
    nbin = n // bin_size

    x_trim = x[:nbin * bin_size]
    y_trim = y[:nbin * bin_size]

    xb = x_trim.reshape(nbin, bin_size).mean(axis=1)
    yb = np.nanmedian(y_trim.reshape(nbin, bin_size), axis=1)

    return xb, yb

def fourier_ripple_model(params, x, n_harmonics=4):
    """
    Generic periodic ripple model.

    params:
        [offset, slope, period, phase0,
         cos1, sin1, cos2, sin2, ..., cosN, sinN]
    """

    offset = params[0]
    slope = params[1]
    period = params[2]
    phase0 = params[3]
    coeffs = params[4:]

    x = np.asarray(x, dtype=float)
    xn = (x - x.mean()) / np.ptp(x)

    phase = 2 * np.pi * (x - phase0) / period

    y = offset + slope * xn

    for k in range(1, n_harmonics + 1):
        ak = coeffs[2 * (k - 1)]
        bk = coeffs[2 * (k - 1) + 1]

        y += ak * np.cos(k * phase)
        y += bk * np.sin(k * phase)

    return y

def fit_fourier_ripple_fast(
    x,
    y,
    period_guess,
    n_harmonics=4,
    bin_size=1024,
):
    """
    Fit the ripple on a binned version of a very long spectrum.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xb, yb = bin_spectrum(x, y, bin_size=bin_size)

    good = np.isfinite(xb) & np.isfinite(yb)
    xb = xb[good]
    yb = yb[good]

    offset0 = np.nanmedian(yb)
    slope0 = 0.0
    phase0 = xb[np.nanargmax(yb)]

    amp0 = 0.5 * (np.nanmax(yb) - np.nanmin(yb))

    coeffs0 = np.zeros(2 * n_harmonics)
    coeffs0[0] = amp0

    p0 = np.r_[
        offset0,
        slope0,
        period_guess,
        phase0,
        coeffs0,
    ]

    lower = np.r_[
        0.1 * offset0,
        -np.inf,
        0.9 * period_guess,
        xb.min(),
        np.full(2 * n_harmonics, -np.inf),
    ]

    upper = np.r_[
        10.0 * offset0,
        np.inf,
        1.1 * period_guess,
        xb.max(),
        np.full(2 * n_harmonics, np.inf),
    ]

    def resid(params):
        return fourier_ripple_model(params, xb, n_harmonics) - yb

    result = least_squares(
        resid,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.02 * offset0,
        max_nfev=2000,
    )

    return result, xb, yb

def fourier_ripple_model_damped(params, x, n_harmonics=4):
    """
    Generic periodic ripple model with a slowly varying amplitude envelope.

    params:
        [offset, slope, period, phase0, d1, d2,
         cos1, sin1, cos2, sin2, ..., cosN, sinN]

    d1, d2 control the ripple-amplitude envelope.
    """

    offset = params[0]
    slope = params[1]
    period = params[2]
    phase0 = params[3]

    d1 = params[4]
    d2 = params[5]
    d3 = params[6]

    coeffs = params[7:]

    x = np.asarray(x, dtype=float)

    xrange = np.ptp(x)
    if xrange == 0:
        xn = np.zeros_like(x, dtype=float)
    else:
        xn = (x - np.mean(x)) / xrange

    phase = 2 * np.pi * (x - phase0) / period

    ripple = np.zeros_like(x, dtype=float)

    for k in range(1, n_harmonics + 1):
        ak = coeffs[2 * (k - 1)]
        bk = coeffs[2 * (k - 1) + 1]

        ripple += ak * np.cos(k * phase)
        ripple += bk * np.sin(k * phase)

    envelope = np.exp(1.0 + d1 * xn + d2 * xn**2 + d3 * xn**3)

    y = offset + slope * xn + envelope * ripple

    return y

def fit_fourier_ripple_damped(
    x,
    y,
    period_guess,
    n_harmonics=4,
    bin_size=512,
):
    """
    Fit a damped Fourier ripple model to a long spectrum by first binning it.

    This is intended for very long spectra, e.g. 8,388,608 channels.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Bin first, fit only the binned spectrum
    xb, yb = bin_spectrum(x, y, bin_size=bin_size)

    good = np.isfinite(xb) & np.isfinite(yb)
    xb = xb[good]
    yb = yb[good]

    offset0 = np.nanmedian(yb)
    slope0 = 0.0

    # Initial phase guess: location of maximum binned spectrum
    phase0 = xb[np.nanargmax(yb)]

    amp0 = 0.5 * (np.nanmax(yb) - np.nanmin(yb))

    # Damping/envelope guesses
    d1_0 = 0.0
    d2_0 = 0.0
    d3_0 = 0.0

    # Fourier coefficient guesses
    coeffs0 = np.zeros(2 * n_harmonics)
    coeffs0[0] = amp0

    p0 = np.r_[
        offset0,
        slope0,
        period_guess,
        phase0,
        d1_0,
        d2_0,
        d3_0,
        coeffs0,
    ]

    lower = np.r_[
        0.1 * offset0,
        -np.inf,
        0.8 * period_guess,
        xb.min(),
        -10.0,
        -10.0,
        -10.0,
        np.full(2 * n_harmonics, -np.inf),
    ]

    upper = np.r_[
        10.0 * offset0,
        np.inf,
        1.2 * period_guess,
        xb.max(),
        10.0,
        10.0,
        10.0,
        np.full(2 * n_harmonics, np.inf),
    ]

    def resid(params):
        return (
            fourier_ripple_model_damped(
                params,
                xb,
                n_harmonics=n_harmonics,
            )
            - yb
        )

    result = least_squares(
        resid,
        p0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=0.02 * offset0,
        max_nfev=5000,
    )

    return result, xb, yb

def main(TESTDATA, OUTFILE):
    # TESTDATA = '/mnt/d/data1/lwa_seti/059654_002588958-waterfall_tun1_test.h5'
    # TESTDATA = '/mnt/d/data1/lwa_seti/059654_002588958-waterfall_tun2.h5'
    # TESTDATA = '/mnt/d/data1/lwa_seti/059654_003048461-waterfall_tun1.h5'

    # freqs = np.load('/mnt/d/data1/lwa_seti/test_freqs.npy')
    # timestep = np.load('/mnt/d/data1/lwa_seti/test_timestep.npy')
    # spec = np.load('/mnt/d/data1/lwa_seti/test_chunk.npy') #2996000:3038800

    with h5py.File(TESTDATA, 'r') as f:
        # frequency structure
        fch1 = f['data'].attrs['fch1'] #MHz
        foff = f['data'].attrs['foff'] #MHz
        nchans = f['data'].attrs['nchans']
        freqs = fch1 + foff * np.arange(nchans)

        # bandpass edge cut
        edges = int(round(len(freqs) * 0.1))
        print('edges', edges)

        keys = list(f.keys())

    # Instrumental Bandpass Stage
    bpm_i = compute_bandpass_instrumental(freqs*1e6)
    bpm_i = (bpm_i - bpm_i.min()) / (bpm_i.max() - bpm_i.min())
    bpm_i[bpm_i == 0.0] += 1e-16

    ## The Claude Juice
    # Use savgol filter on 32 time samples from the dataset and median them to get a ripple. 
    if 'corrected' not in keys:
        print('Uncorrected File, making the fit!')
        start = time.time()
        bpm2 = computeBandpassData_streaming(
                hdf5_path=TESTDATA,
                input_dataset="data",
                output_dataset="corrected",
                bpm_dataset="bandpass_model",
                bpm_estimation_chunks=32,
                instr_bandpass=bpm_i, 
                window_size = 41
            )
        print(f"total seconds = {np.abs(time.time()-start)}") # This took 194.83 seconds on my laptop
    else:
        print('Corrected File, proceed!')

    ### Pull out the bandpass model from the filter
    with h5py.File(TESTDATA, 'r') as f:
        # Get smoothed model of the bandpass ripple 
        bpm_m = f['bandpass_model'][:]
        n_rows = f['data'].shape[0]
        row_idx = min(100, n_rows-1)
        data = f['data'][row_idx,0,:].astype(np.float32)
        # code_corrected = f['corrected'][row_idx,0,:].astype(np.float32)

    ### Here we are trying a generic fourier series fit
    ## Use only the interior region of the bandpass that we would like to keep. 
    x_full = np.arange(len(freqs))
    x_inner = np.arange(len(bpm_m))
    result, xb, yb = fit_fourier_ripple_damped(x_inner, bpm_m, period_guess=10700, n_harmonics=5, bin_size=256)

    ### Build a model of the ripple, then pad it out to the proper size. {This gets dropped later} 
    y_model1 = fourier_ripple_model_damped(result.x, x_inner, n_harmonics=5)
    y_model2 = np.pad(y_model1, pad_width = 0, mode='constant', constant_values=np.mean(y_model1))

    ### Divide out the model fit
    data_corr = data / (bpm_i * y_model2)

    ### Write model
    full_model = (bpm_i * y_model2)
    # full_model.tofile(OUTFILE)

    ### savgol
    bpm5 = smooth_bpmodel(bpm_m, od=4)
    bpm5 = bpm5.data.astype(np.float32)
    bpm5.tofile(OUTFILE) #+'.savgol')
    """
    ### Plotting:
    fig = plt.figure()
    plt.plot(data_corr[:], 'k')
    plt.ylim(np.median(data_corr[:]) - 1000, np.median(data_corr[:]) + 1000)
    plt.show()

    fig = plt.figure()
    plt.plot(x_full, (bpm_i * y_model2), 'cyan')
    plt.show()
    """
    return
    
if __name__ == "__main__":
    filename = sys.argv[-1]
    outfile = os.path.basename(filename).split('.')[0] + "_bpmodel.f32"

    main(filename, outfile)

    ## Lillian test data
    # /data/local/lekness/scripts_lillian/*_tun[1,2].h5

#make 4-tap hamming window, invert it, multiply it to the DRX data, save inner ~85%
#ndp t-engine

### How I found the channel width of ripples:
## Compute autocorrelation, split in middle at zero point, find peaks in figure (or analytically)
# timestep_m = timestep - np.nanmedian(timestep)
# ac = correlate(timestep_m, timestep_m, 'full')
# ac2 = ac[len(ac)//2:]

# fig = plt.figure()
# plt.plot(ac2)
# plt.show()
