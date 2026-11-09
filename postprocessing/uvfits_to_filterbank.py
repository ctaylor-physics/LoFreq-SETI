"""Convert UVFITS cross-correlation visibilities to BLISS filterbank HDF5.

Each output contains one baseline, one polarization, one scan, and one
uniformly spaced IF window. Samples are uncalibrated visibility amplitudes.
The default polarization index 0 is the XX parallel hand in the sample file's
AIPS AN receptor order; its UVFITS STOKES code is retained separately.
"""
import argparse
from pathlib import Path
import re

from astropy.io import fits
import h5py
import numpy as np


def describe(hdul):
    hdu = hdul[0]
    data = hdu.data
    if data is None or data.data.ndim != 7 or data.data.shape[1:3] != (1, 1) or data.data.shape[-1] != 3:
        raise ValueError('Expected UVFITS random groups with singleton RA/DEC and REAL/IMAG/WEIGHT axes')
    if hdu.header.get('NAXIS4') != 1:
        raise ValueError('Only the current one-frequency-channel-per-IF layout (NAXIS4=1) is supported')
    if 'AIPS FQ' not in hdul or len(hdul['AIPS FQ'].data) != 1:
        raise ValueError('Expected exactly one AIPS FQ frequency setup')
    fq = hdul['AIPS FQ'].data[0]
    freq_ref = float(hdu.header['CRVAL4']) + (1 - float(hdu.header.get('CRPIX4', 1))) * float(hdu.header.get('CDELT4', 0))
    freq = freq_ref + np.atleast_1d(np.asarray(fq['IF FREQ'], dtype='f8'))
    widths = np.atleast_1d(np.asarray(fq['CH WIDTH'], dtype='f8'))
    sidebands = np.atleast_1d(np.asarray(fq['SIDEBAND'], dtype='i4'))
    if not (len(freq) == len(widths) == len(sidebands) == hdu.header['NAXIS5']):
        raise ValueError('AIPS FQ and IF axis disagree')
    if len(freq) < 2 or not np.all(np.isfinite(freq)) or np.any(np.diff(freq) <= 0):
        raise ValueError('Expected at least two strictly increasing IF frequencies')
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0) or np.any(sidebands != 1):
        raise ValueError('Only positive-width, upper-sideband IFs are supported')
    cuts = [0] + [i + 1 for i, delta in enumerate(np.diff(freq))
                  if not np.isclose(delta, widths[i], rtol=1e-5, atol=0.1)] + [len(freq)]
    windows = []
    for start, stop in zip(cuts[:-1], cuts[1:]):
        block = freq[start:stop]
        if len(block) < 3:
            raise ValueError(f'IF window beginning at {start} has fewer than three channels required by the filterbank reader')
        if not np.allclose(np.diff(block), block[1] - block[0], rtol=1e-5, atol=0.1):
            raise ValueError('Nonlinear frequency window')
        windows.append((start, stop, block / 1e6))
    codes = np.asarray(data.par('BASELINE'), dtype='f8')
    if not np.all(np.isfinite(codes)) or np.any(codes != np.floor(codes)):
        raise ValueError('Invalid baseline codes')
    baselines = np.unique(codes.astype('i8')).tolist()
    npol = data.data.shape[-2]
    stokes = [int(round(hdu.header.get('CRVAL3', 1) + i * hdu.header.get('CDELT3', 1)))
              for i in range(npol)]
    return baselines, stokes, windows


def select(option, available, label):
    if option == 'all':
        return list(available)
    try:
        chosen = [int(item) for item in option.split(',')]
    except ValueError as exc:
        raise ValueError(f'{label}: use "all" or comma-separated integers') from exc
    if not chosen or len(set(chosen)) != len(chosen) or not set(chosen) <= set(available):
        raise ValueError(f'{label}: select distinct values from {list(available)}')
    return chosen


def scans(jd, integrations, nx_numbers=None):
    if len(jd) < 2 or not np.all(np.isfinite(jd)) or np.any(np.diff(jd) <= 0):
        raise ValueError('Baseline timestamps must be finite, unique, and increasing')
    if not np.all(np.isfinite(integrations)) or np.any(integrations <= 0):
        raise ValueError('Invalid integration times')
    tsamp = float(np.median(integrations))
    separations = np.diff(jd) * 86400
    if np.any(separations < 0.5 * tsamp):
        raise ValueError('Overlapping integrations cannot be represented by one filterbank time axis')
    breaks = separations > 1.5 * tsamp
    if nx_numbers is not None:
        breaks |= np.diff(nx_numbers) != 0
    boundaries = np.r_[0, np.flatnonzero(breaks) + 1, len(jd)]
    parts = [(int(start), int(stop)) for start, stop in zip(boundaries[:-1], boundaries[1:])]
    if any(stop - start < 2 for start, stop in parts):
        raise ValueError('A scan has fewer than two integrations')
    return parts, tsamp


def filterbank_coordinates(header):
    ra_deg = float(header.get('OBSRA', header.get('CRVAL6', 0)))
    dec_deg = float(header.get('OBSDEC', header.get('CRVAL7', 0)))
    if not np.isfinite(ra_deg) or not np.isfinite(dec_deg) or abs(dec_deg) > 90:
        raise ValueError('Invalid source coordinates')
    # blimpy's HDF5 reader interprets these as decimal hours and degrees.
    return (ra_deg % 360) / 15, dec_deg, ra_deg, dec_deg


def nx_scan_numbers(hdul):
    """Map zero-based visibility rows to AIPS NX scans when the table is valid."""
    if 'AIPS NX' not in hdul:
        return None
    nx = hdul['AIPS NX'].data
    starts = np.asarray(nx['START VIS'], dtype='i8')
    ends = np.asarray(nx['END VIS'], dtype='i8')
    if (len(starts) == 0 or starts[0] != 1 or ends[-1] != len(hdul[0].data)
            or np.any(starts[1:] != ends[:-1] + 1) or np.any(ends < starts)):
        raise ValueError('AIPS NX scan ranges do not cover the visibility records')
    return np.searchsorted(ends, np.arange(1, len(hdul[0].data) + 1), side='left')


def assumed_receptor_pair(hdul, baseline, pol):
    """Label the product from AIPS AN receptors; keep FITS Stokes separately."""
    if 'AIPS AN' not in hdul or pol >= 4 or baseline >= 65536:
        return 'UNKNOWN'
    antennas = {int(row['NOSTA']): (str(row['POLTYA']).strip(), str(row['POLTYB']).strip())
                for row in hdul['AIPS AN'].data}
    first = antennas.get(baseline // 256)
    second = antennas.get(baseline % 256)
    if first is None or second is None:
        return 'UNKNOWN'
    a, b = ((0, 0), (1, 1), (0, 1), (1, 0))[pol]
    return first[a] + second[b]


def write_filterbank(path, raw, jd, freq, header, baseline, pol, stokes,
                     scan, band, tsamp, source_file, measure, receptor_pair):
    weights = raw[..., 2]
    amplitude = np.hypot(raw[..., 0], raw[..., 1])
    values = amplitude if measure == 'amplitude' else amplitude * amplitude
    bad = ~np.isfinite(values) | ~np.isfinite(weights) | (weights <= 0)
    values = np.where(bad, 0.0, values).astype('f4')
    src_raj, src_dej, ra_deg, dec_deg = filterbank_coordinates(header)
    with h5py.File(path, 'w') as f:
        f.attrs['CLASS'], f.attrs['VERSION'] = 'FILTERBANK', '1.0'
        ds = f.create_dataset('data', data=values[:, None, :])
        mask = f.create_dataset('mask', data=bad[:, None, :].astype('u1'))
        mask.attrs['description'] = '1 = nonpositive/invalid visibility weight or nonfinite data'
        for dataset in (ds, mask):
            for axis, name in enumerate(('time', 'feed_id', 'frequency')):
                dataset.dims[axis].label = name
        f.create_dataset('time_mjd', data=jd - 2400000.5)
        f.create_dataset('frequency_mhz', data=freq)
        f.create_dataset('visibility_weight', data=weights.astype('f4'))
        ds.attrs.update({
            'machine_id': 0, 'telescope_id': -1, 'data_type': 1,
            'fch1': float(freq[0]), 'foff': float(freq[1] - freq[0]),
            'nchans': len(freq), 'nifs': 1, 'nbeams': 1, 'ibeam': -1, 'nbits': 32,
            'tstart': float(jd[0] - 2400000.5), 'tsamp': tsamp,
            'src_raj': src_raj, 'src_dej': src_dej,
            'az_start': 0, 'za_start': 0,
            'source_name': str(header.get('OBJECT', 'UNKNOWN')),
            'rawdatafile': source_file, 'uvfits_baseline': int(baseline),
            'uvfits_antenna1': int(baseline) // 256, 'uvfits_antenna2': int(baseline) % 256,
            'uvfits_polarization_index': int(pol), 'uvfits_stokes_code': int(stokes),
            'uvfits_scan_number': scan, 'uvfits_window_number': band,
            'uvfits_measure': measure, 'uvfits_bunit': str(header.get('BUNIT', 'UNKNOWN')),
            'uvfits_receptor_pair_assumed': receptor_pair,
            'uvfits_ra_deg': ra_deg, 'uvfits_dec_deg': dec_deg,
        })


def convert(input_path, outdir, baselines='all', polarizations='0', measure='amplitude',
            force=False, scan_selection='all', window_selection='all'):
    """Write one HDF5 per baseline, polarization, continuous scan, and IF window."""
    source, outdir = Path(input_path).resolve(), Path(outdir).resolve()
    if measure not in ('amplitude', 'power'):
        raise ValueError('measure must be amplitude or power')
    with fits.open(source, memmap=True) as hdul:
        available_baselines, stokes, windows = describe(hdul)
        chosen_baselines = select(baselines, available_baselines, 'baselines')
        chosen_pols = select(polarizations, range(len(stokes)), 'polarizations')
        chosen_windows = select(window_selection, range(1, len(windows) + 1), 'windows')
        groups = hdul[0].data
        baseline_codes = np.asarray(groups.par('BASELINE'), dtype='i8')
        dates = np.asarray(groups.par('DATE'), dtype='f8')
        integrations = np.asarray(groups.par('INTTIM'), dtype='f8')
        nx_numbers = nx_scan_numbers(hdul)
        jobs = []
        for baseline in chosen_baselines:
            rows = np.flatnonzero(baseline_codes == baseline)
            rows = rows[np.argsort(dates[rows], kind='stable')]
            segments, tsamp = scans(dates[rows], integrations[rows],
                                    None if nx_numbers is None else nx_numbers[rows])
            chosen_scans = select(scan_selection, range(1, len(segments) + 1), 'scans')
            for scan_number, (start, stop) in enumerate(segments, 1):
                if scan_number not in chosen_scans:
                    continue
                for band_number, (first, last, freq) in enumerate(windows, 1):
                    if band_number not in chosen_windows:
                        continue
                    for pol in chosen_pols:
                        filename = (f'{re.sub(r"[^A-Za-z0-9_.-]", "_", source.name)}_'
                                    f'bl{baseline}_p{pol}_scan{scan_number:03d}_band{band_number:02d}.h5')
                        jobs.append((outdir / filename, rows[start:stop], freq, first, last,
                                     baseline, pol, scan_number, band_number, tsamp))
        if not force:
            existing = [path for path, *_ in jobs if path.exists()]
            if existing:
                raise FileExistsError(f'{existing[0]} exists; use --force to replace outputs')
        outdir.mkdir(parents=True, exist_ok=True)
        for path, rows, freq, first, last, baseline, pol, scan_number, band_number, tsamp in jobs:
            block = np.asarray(groups.data[rows, 0, 0, first:last, 0, pol, :])
            tmp = path.with_name('.' + path.name + '.partial')
            try:
                write_filterbank(tmp, block, dates[rows], freq, hdul[0].header,
                                 baseline, pol, stokes[pol], scan_number, band_number,
                                 tsamp, source.name, measure,
                                 assumed_receptor_pair(hdul, baseline, pol))
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
        return [job[0] for job in jobs]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('uvfits')
    parser.add_argument('--outdir', default='uvfits_filterbanks')
    parser.add_argument('--baselines', default='all', help='AIPS baseline code(s), comma-separated or all')
    parser.add_argument('--polarizations', default='0', help='Zero-based polarization index(es), comma-separated or all (default: 0, XX for sample file)')
    parser.add_argument('--scans', default='all', help='One-based scan number(s), comma-separated or all')
    parser.add_argument('--windows', default='all', help='One-based frequency-window number(s), comma-separated or all')
    parser.add_argument('--measure', choices=('amplitude', 'power'), default='amplitude')
    parser.add_argument('--list', action='store_true', help='Inspect the file without converting it')
    parser.add_argument('--force', action='store_true', help='Replace existing outputs')
    args = parser.parse_args()
    with fits.open(args.uvfits, memmap=True) as hdul:
        baselines, stokes, windows = describe(hdul)
        if 'AIPS AN' in hdul:
            names = {int(number): str(name).strip() for number, name in
                     zip(hdul['AIPS AN'].data['NOSTA'], hdul['AIPS AN'].data['ANNAME'])}
            print('Baselines:', [(code, names.get(code // 256, str(code // 256)),
                                  names.get(code % 256, str(code % 256))) for code in baselines])
        else:
            print('Baselines:', baselines)
        print('Polarization indices and FITS Stokes codes:', list(enumerate(stokes)))
        if 'AIPS NX' in hdul:
            print('AIPS NX scans:', len(hdul['AIPS NX'].data))
        for i, (first, last, freq) in enumerate(windows, 1):
            print(f'Window {i}: IF [{first}, {last}), {freq[0]:.6f}–{freq[-1]:.6f} MHz')
    if not args.list:
        outputs = convert(args.uvfits, args.outdir, args.baselines, args.polarizations,
                          args.measure, args.force, args.scans, args.windows)
        print(f'Wrote {len(outputs)} filterbank HDF5 files in {Path(args.outdir).resolve()}')


if __name__ == '__main__':
    main()
