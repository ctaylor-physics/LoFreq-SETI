"""Fit and apply a bandpass on the valid interior of a full-grid LWA tuning.

The final smoothed float32 profile is stored in ``bandpass_model`` and exported
unchanged to ``<basename>_bpmodel.f32``. Original data become ``uncorrected``;
flattened data become ``data``. Excluded edges stay -1.0 and the mask is retained.
The old chunk-and-BLISS runner must not be used on these already corrected files.
"""
import argparse
import os
import operator

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from lsl.common import stations, ndp
from lsl.reader.drx import FILTER_CODES

try:
    import cupy as cp
except ImportError:
    cp = None


def array_backend():
    """Use CuPy when a device is accessible, otherwise keep processing on CPU."""
    if cp is not None:
        try:
            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp
        except cp.cuda.runtime.CUDARuntimeError:
            pass
    return np


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


# Only this group is owned by an in-progress correction. It holds hard links
# during promotion, so an interrupted dataset rename never loses the raw input.
WORK_GROUP = '_bandpass_work'
CORRECTION_VERSION = 1
LAYOUT_KEYS = ('coarse_layout_version', 'num_coarse', 'edge_coarse',
               'fine_channels_per_coarse', 'valid_channel_start', 'valid_channel_stop',
               'edge_mask_value')


def read_layout(ds, mask):
    """Validate the first-half metadata without inferring or changing a layout."""
    if ds.ndim != 3 or ds.shape[1] != 1 or ds.shape[0] == 0 or ds.shape[2] < 2:
        raise ValueError('Expected nonempty tuning data with shape (time, 1, frequency)')
    missing = [key for key in LAYOUT_KEYS + ('nchans', 'fch1', 'foff') if key not in ds.attrs]
    if missing:
        raise ValueError(f'Missing full-grid metadata: {missing}; regenerate with the updated first half')
    values = {}
    for key in LAYOUT_KEYS[:-1] + ('nchans',):
        try:
            values[key] = operator.index(ds.attrs[key])
        except TypeError as exc:
            raise ValueError(f'{key} must be an integer') from exc
    n = ds.shape[2]
    coarse, edge = values['num_coarse'], values['edge_coarse']
    if values['coarse_layout_version'] != 1 or values['nchans'] != n:
        raise ValueError('Unsupported layout version or nchans does not match the dataset')
    if coarse <= 0 or edge < 0 or 2 * edge >= coarse or n % coarse:
        raise ValueError('Invalid coarse-channel count, divisibility, or edge exclusion')
    width = n // coarse
    start, stop = edge * width, n - edge * width
    if (values['fine_channels_per_coarse'], values['valid_channel_start'],
            values['valid_channel_stop']) != (width, start, stop):
        raise ValueError('Stored coarse-channel layout is inconsistent')
    if ds.attrs['edge_mask_value'] != -1.0:
        raise ValueError('Expected edge_mask_value = -1.0')
    if not np.isfinite(ds.attrs['fch1']) or not np.isfinite(ds.attrs['foff']) or ds.attrs['foff'] == 0:
        raise ValueError('Frequency origin and spacing must be finite, with nonzero spacing')
    if mask.shape != ds.shape or mask.dtype != np.dtype('uint8'):
        raise ValueError('Expected a uint8 mask with the same shape as data')
    return start, stop


def positive_profile(profile, label, channel_offset=0):
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError(f'{label} must be a nonempty one-dimensional profile')
    nonfinite = ~np.isfinite(profile)
    nonpositive = np.isfinite(profile) & (profile <= 0)
    if np.any(nonfinite | nonpositive):
        channels = np.flatnonzero(nonfinite | nonpositive)[:8] + channel_offset
        finite = profile[np.isfinite(profile)]
        minimum = float(finite.min()) if finite.size else float('nan')
        raise ValueError(
            f'{label} must be finite and positive throughout the valid interval: '
            f'{np.count_nonzero(nonfinite)} nonfinite, '
            f'{np.count_nonzero(nonpositive)} nonpositive of {profile.size}; '
            f'finite minimum={minimum:.6g}; first affected fine-channel indices={channels.tolist()}'
        )
    return profile


def smoothing_parameters(size, window, order):
    if size < 1 or window < 1 or order < 0:
        raise ValueError('Smoothing needs a nonempty interval, positive window, and nonnegative order')
    window = min(int(window), size)
    if window % 2 == 0:
        window -= 1
    return window, min(order, window - 1)


def smooth_positive(profile, window, order, label, channel_offset=0):
    """Keep the linear SG fit when valid; otherwise fit positive data in log space.

    Polynomial filters have negative weights and can undershoot around strong
    narrow features. Bound the fallback log fit to the observed input range,
    avoiding artificial near-zero denominators from polynomial extrapolation.
    Invalid input values are diagnosed, never silently clipped or interpolated.
    """
    profile = positive_profile(profile, f'{label} input', channel_offset)
    result = savgol_filter(profile, window, order)
    if np.all(np.isfinite(result)) and np.all(result > 0):
        return result, 'linear'
    bad = ~np.isfinite(result) | (result <= 0)
    print(f'{label}: linear smoothing produced {np.count_nonzero(bad)} invalid bins; '
          'using bounded log-space smoothing of the positive input')
    logarithm = np.log(profile)
    fitted_log = savgol_filter(logarithm, window, order)
    if not np.all(np.isfinite(fitted_log)):
        raise ValueError(f'{label}: log-space smoothing produced nonfinite values')
    result = np.exp(np.clip(fitted_log, logarithm.min(), logarithm.max()))
    return positive_profile(result, label, channel_offset), 'log_fallback'


def smooth_bpmodel(bpmodel, od=4, window_size=None, *, channel_offset=0, return_method=False):
    """Final smoothing of the valid interval only; return a mean-one profile."""
    profile = positive_profile(bpmodel, 'Combined bandpass', channel_offset)
    requested = window_size if window_size is not None else (int(round(np.sqrt(profile.size))) | 1)
    window, order = smoothing_parameters(profile.size, requested, od)
    result, method = smooth_positive(profile, window, order, 'Smoothed bandpass', channel_offset)
    result = result / result.mean()
    return (result, method) if return_method else result


def _promote(f):
    """Idempotently finish a validated correction using staged hard links."""
    work = f[WORK_GROUP]
    if work.attrs.get('state') != 'ready':
        raise ValueError('Incomplete bandpass correction; rerun with --force to restart from raw data')
    raw, corrected, model = work['raw'], work['corrected'], work['bandpass_model']
    start, stop = read_layout(raw, f['mask'])
    read_layout(corrected, f['mask'])
    if corrected.shape != raw.shape or model.shape != (raw.shape[2],):
        raise ValueError('Staged bandpass output has inconsistent shapes')
    if corrected.attrs.get('bandpass_correction_version') != CORRECTION_VERSION:
        raise ValueError('Staged corrected data are not marked complete')
    positive_profile(model[start:stop], 'Staged bandpass', start)
    if 'uncorrected' in f and f['uncorrected'].id != raw.id:
        raise ValueError('Existing uncorrected data do not match the staged source')
    if 'uncorrected' not in f:
        f['uncorrected'] = raw
    for name, dataset in (('data', corrected), ('bandpass_model', model)):
        if name in f and f[name].id == dataset.id:
            continue
        if name in f:
            del f[name]
        f[name] = dataset
    f.flush()
    del f[WORK_GROUP]
    f.flush()


def _source(f, force):
    if WORK_GROUP in f:
        if f[WORK_GROUP].attrs.get('state') == 'ready':
            _promote(f)
        elif not force:
            raise ValueError('Incomplete bandpass correction; rerun with --force to restart from raw data')
        else:
            # A writing-stage failure has not changed the public datasets.
            work = f[WORK_GROUP]
            if work.attrs.get('state') != 'writing':
                raise ValueError('Unknown bandpass work state; refusing to discard it')
            source = f.get('uncorrected', f.get('data'))
            if source is None or ('raw' in work and source.id != work['raw'].id):
                raise ValueError('Cannot safely restart: original data are not available')
            del f[WORK_GROUP]
    if 'corrected' in f:
        raise ValueError('Legacy corrected dataset found; use a fresh full-grid first-half file')
    if 'data' not in f or 'mask' not in f:
        raise ValueError('Expected data and mask datasets')
    complete = f['data'].attrs.get('bandpass_correction_version')
    if complete is not None:
        if complete != CORRECTION_VERSION or 'uncorrected' not in f or 'bandpass_model' not in f:
            raise ValueError('Unrecognized or incomplete bandpass correction')
        ds = f['uncorrected']
        start, stop = read_layout(ds, f['mask'])
        if read_layout(f['data'], f['mask']) != (start, stop) or f['data'].shape != ds.shape:
            raise ValueError('Corrected and original layouts do not match')
        model = f['bandpass_model']
        if model.shape != (ds.shape[2],) or model.attrs.get('bandpass_correction_version') != CORRECTION_VERSION:
            raise ValueError('Invalid saved bandpass model')
        if not force:
            positive_profile(model[start:stop], 'Saved bandpass', start)
            print('Bandpass correction already complete; reusing the saved profile')
            return None
        return ds
    if 'uncorrected' in f or 'bandpass_model' in f:
        raise ValueError('Ambiguous previous correction; use a fresh full-grid first-half file')
    return f['data']


def _correct_blocks(ds, mask, out, profile, start, stop, rows_per_block, channel_block, xp=np):
    """Check mask/sentinels and correct in bounded time-frequency blocks."""
    for row in range(0, ds.shape[0], rows_per_block):
        end = min(row + rows_per_block, ds.shape[0])
        for left, right, valid in ((0, start, False), (start, stop, True),
                                   (stop, ds.shape[2], False)):
            for col in range(left, right, channel_block):
                last = min(col + channel_block, right)
                selection = np.s_[row:end, 0, col:last]
                flags = mask[selection]
                if np.any(flags != (0 if valid else 1)):
                    raise ValueError('Mask does not match the stored valid-channel interval')
                data = ds[selection]
                if not valid:
                    if np.any(data != -1.0):
                        raise ValueError('Excluded edge samples must equal -1.0')
                    continue  # The output fill value is -1.0.
                if not np.all(np.isfinite(data)):
                    raise ValueError('Nonfinite data in the valid interval')
                with np.errstate(over='ignore', invalid='ignore'):
                    corrected = (xp.asarray(data, dtype=xp.float64) /
                                 xp.asarray(profile[col:last])).astype(xp.float32)
                    if xp is not np:
                        corrected = xp.asnumpy(corrected)
                if not np.all(np.isfinite(corrected)):
                    raise ValueError('Bandpass correction produced nonfinite float32 values')
                out[selection] = corrected
        print(f'Corrected rows {end}/{ds.shape[0]}')


def computeBandpassData_streaming(
    hdf5_path, input_dataset='data', output_dataset='corrected',
    bpm_dataset='bandpass_model', chunk_size=16, bpm_estimation_chunks=32,
    instr_bandpass=None, window_size=41, *, force=False, channel_block=131072,
    final_window=None, use_instrumental=False,
):
    """Apply the final smoothed profile and promote the result, preserving raw data.

    Input/output names are retained for existing callers, but the public schema
    is fixed. Forced refits always use uncorrected, never the flattened data.
    An optional supplied instrumental model is full-length; only its interior
    is used. Otherwise use_instrumental computes the LSL model on that interior.
    """
    if (input_dataset, output_dataset, bpm_dataset) != ('data', 'corrected', 'bandpass_model'):
        raise ValueError('Expected dataset names data, corrected, and bandpass_model')
    if min(chunk_size, bpm_estimation_chunks, channel_block, window_size) < 1:
        raise ValueError('Block sizes, estimation-row count, and smoothing window must be positive')
    if final_window is not None and final_window < 1:
        raise ValueError('Final smoothing window must be positive')
    with h5py.File(hdf5_path, 'r+') as f:
        ds = _source(f, force)
        if ds is None:
            return f['bandpass_model'][:]
        start, stop = read_layout(ds, f['mask'])
        nrows, _, nchans = ds.shape
        size = stop - start
        xp = array_backend()
        print(f'Backend: {"CPU" if xp is np else "GPU"}')
        print(f'Fitting valid fine channels [{start}, {stop}) of {nchans}')
        instrument = None
        if instr_bandpass is not None:
            supplied = np.asarray(instr_bandpass)
            if supplied.shape != (nchans,):
                raise ValueError('Instrumental bandpass must match the full fine-channel count')
            instrument = supplied[start:stop]
        elif use_instrumental:
            freqs = ds.attrs['fch1'] + ds.attrs['foff'] * np.arange(start, stop)
            instrument = compute_bandpass_instrumental(freqs * 1e6)
        if instrument is not None:
            instrument = positive_profile(instrument, 'Instrumental bandpass', start)
            if instrument.shape != (size,):
                raise ValueError('Instrumental response does not match the valid interval')
            # Do not subtract the minimum: a response floor near zero can amplify
            # data arbitrarily. Fail clearly for invalid/unsupported responses.
            instrument = instrument / instrument.mean()

        sample_idx = np.linspace(0, nrows - 1, min(bpm_estimation_chunks, nrows), dtype=int)
        median = np.empty(size, dtype=np.float64)
        for offset in range(0, size, channel_block):
            end = min(offset + channel_block, size)
            samples = np.empty((len(sample_idx), end - offset), dtype=np.float64)
            for i, row in enumerate(sample_idx):
                samples[i] = ds[row, 0, start + offset:start + end]
            if not np.all(np.isfinite(samples)):
                raise ValueError('Nonfinite sampled data in the valid interval')
            if instrument is not None:
                samples /= instrument[offset:end]
            values = xp.median(xp.asarray(samples), axis=0)
            median[offset:end] = values if xp is np else xp.asnumpy(values)
        window, order = smoothing_parameters(size, min(window_size, max(1, round(size / 10))), 9)
        residual, residual_method = smooth_positive(median, window, order, 'Residual bandpass', start)
        residual /= residual.mean()
        combined = residual if instrument is None else instrument * residual
        final_ws, final_od = smoothing_parameters(
            size, final_window if final_window is not None else (int(round(np.sqrt(size))) | 1), 4)
        final, final_method = smooth_bpmodel(combined, od=final_od, window_size=final_ws,
                                             channel_offset=start, return_method=True)
        profile = np.ones(nchans, dtype=np.float32)
        profile[start:stop] = final.astype(np.float32)
        positive_profile(profile[start:stop], 'Final float32 bandpass', start)

        work = f.create_group(WORK_GROUP)
        work.attrs['state'] = 'writing'
        work['raw'] = ds
        model = work.create_dataset('bandpass_model', data=profile)
        metadata = {
            'bandpass_correction_version': CORRECTION_VERSION,
            'source_dataset': 'uncorrected',
            'bpm_dataset': 'bandpass_model',
            'bandpass_correction': 'savgol_median_final_smoothed',
            'savgol_window': window, 'savgol_order': order,
            'final_savgol_window': final_ws, 'final_savgol_order': final_od,
            'smoothing_policy': 'linear_with_bounded_log_fallback_v1',
            'residual_smoothing_method': residual_method,
            'final_smoothing_method': final_method,
            'estimation_rows_used': len(sample_idx),
            'instrumental_bandpass': instrument is not None,
        }
        model.attrs.update(metadata)
        for key in LAYOUT_KEYS:
            model.attrs[key] = ds.attrs[key]
        model.attrs['excluded_region_value'] = 1.0
        out = work.create_dataset('corrected', shape=ds.shape, dtype=np.float32,
                                  chunks=(1, 1, min(nchans, channel_block)), fillvalue=-1.0)
        out.attrs.update(dict(ds.attrs))
        # Completion version is set only after all data blocks pass validation.
        out.attrs.update({key: value for key, value in metadata.items()
                          if key != 'bandpass_correction_version'})
        out.attrs['nbits'] = 32
        for axis, dim in enumerate(ds.dims):
            out.dims[axis].label = dim.label
        f.flush()
        _correct_blocks(ds, f['mask'], out, profile, start, stop, chunk_size, channel_block, xp)
        out.attrs['bandpass_correction_version'] = CORRECTION_VERSION
        f.flush()
        work.attrs['state'] = 'ready'
        f.flush()
        _promote(f)
        print('Saved final profile and promoted flattened data; originals retained as uncorrected')
        return profile


def main(TESTDATA, OUTFILE=None, *, force=False, **options):
    if OUTFILE is None:
        OUTFILE = os.path.basename(TESTDATA).split('.')[0] + '_bpmodel.f32'
    if os.path.realpath(OUTFILE) == os.path.realpath(TESTDATA):
        raise ValueError('Profile export must not overwrite the input HDF5 file')
    profile = computeBandpassData_streaming(TESTDATA, force=force, use_instrumental=True, **options)
    # The saved and applied profile is already float32; no additional smoothing.
    profile.tofile(OUTFILE)
    print(f'Exported the applied profile to {OUTFILE}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('h5_path')
    parser.add_argument('--output', help='Export profile path (default: <basename>_bpmodel.f32 in cwd)')
    parser.add_argument('--force', action='store_true', help='Refit original data; restart interrupted correction')
    parser.add_argument('--sample-rows', type=int, default=32)
    parser.add_argument('--row-block', type=int, default=16)
    parser.add_argument('--channel-block', type=int, default=131072)
    parser.add_argument('--window-size', type=int, default=41, help='Residual smoothing window cap')
    parser.add_argument('--final-window', type=int, help='Final smoothing window (default: sqrt(valid channels))')
    args = parser.parse_args()
    main(args.h5_path, args.output, force=args.force, bpm_estimation_chunks=args.sample_rows,
         chunk_size=args.row_block, channel_block=args.channel_block,
         window_size=args.window_size, final_window=args.final_window)
