"""Search corrected full-grid HDF5 directly, without intermediate HDF5 chunks."""
import argparse
import hashlib
import json
import operator
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import h5py
import numpy as np

if __package__:
    from .hits_io import read_dat
else:
    from hits_io import read_dat


def search_layout(path):
    with h5py.File(path, 'r') as f:
        if '_bandpass_work' in f:
            raise ValueError('Incomplete bandpass correction; finish the fitter before searching')
        if not {'data', 'mask', 'uncorrected', 'bandpass_model'} <= set(f):
            raise ValueError('Run the updated bandpass fitter before searching')
        ds = f['data']
        if ds.ndim != 3 or ds.shape[1] != 1 or ds.shape[0] < 2:
            raise ValueError('Expected data shaped (time >= 2, 1, frequency)')
        keys = ('coarse_layout_version', 'num_coarse', 'edge_coarse', 'fine_channels_per_coarse',
                'valid_channel_start', 'valid_channel_stop', 'nchans', 'bandpass_correction_version')
        try:
            a = {key: operator.index(ds.attrs[key]) for key in keys}
        except (KeyError, TypeError) as exc:
            raise ValueError('Missing or invalid full-grid correction metadata') from exc
        n, coarse, edge = ds.shape[2], a['num_coarse'], a['edge_coarse']
        if a['coarse_layout_version'] != 1 or a['bandpass_correction_version'] != 1:
            raise ValueError('Unsupported layout or correction version')
        if n < 2 or coarse < 1 or edge < 0 or 2 * edge >= coarse or n % coarse:
            raise ValueError('Invalid coarse-channel layout')
        width = n // coarse
        if (a['nchans'], a['fine_channels_per_coarse'], a['valid_channel_start'], a['valid_channel_stop']) != (
                n, width, edge * width, n - edge * width):
            raise ValueError('Layout metadata does not match the actual data shape')
        if f['mask'].shape != ds.shape or f['uncorrected'].shape != ds.shape or f['bandpass_model'].shape != (n,):
            raise ValueError('Inconsistent corrected file datasets')
        if f['bandpass_model'].attrs.get('bandpass_correction_version') != 1:
            raise ValueError('Incomplete saved bandpass model')
        if ds.attrs.get('edge_mask_value') != -1.0:
            raise ValueError('Unknown edge sentinel')
        for key in ('fch1', 'foff', 'tsamp'):
            if key not in ds.attrs or not np.isfinite(ds.attrs[key]):
                raise ValueError(f'Missing or invalid {key}')
        if ds.attrs['foff'] == 0 or ds.attrs['tsamp'] <= 0:
            raise ValueError('Frequency spacing must be nonzero and sample interval positive')
        return {'fine_channels_per_coarse': width, 'first_coarse': edge,
                'number_coarse': coarse - 2 * edge}


def file_identity(path):
    stat = Path(path).stat()
    return {'path': str(Path(path).resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def add_search_arguments(parser):
    parser.add_argument('--bliss-executable', default=os.environ.get('BLISS_FIND_HITS', 'bliss_find_hits'),
                        help='bliss_find_hits executable (default: BLISS_FIND_HITS or PATH)')
    parser.add_argument('--device', default=None, help='BLISS device, e.g. cuda:0 or cpu (default: BLISS default)')
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--min-drift', type=float, default=-3.0, help='Minimum drift in Hz/s')
    parser.add_argument('--max-drift', type=float, default=3.0, help='Maximum drift in Hz/s')
    parser.add_argument('--drift-step', type=float, default=1.0, help='Drift-resolution step multiplier')
    parser.add_argument('--distance', type=int, default=7)


def run_search(h5_path, outdir, bliss_executable='bliss_find_hits', device=None,
               snr=10.0, min_drift=-3.0, max_drift=3.0, drift_step=1.0, distance=7, force=False):
    h5_path, outdir = Path(h5_path).resolve(), Path(outdir).resolve()
    if not np.all(np.isfinite([snr, min_drift, max_drift, drift_step])) or snr <= 0 or drift_step <= 0 or min_drift >= max_drift or distance < 1:
        raise ValueError('Invalid BLISS search limits, threshold, or distance')
    layout = search_layout(h5_path)
    executable = shutil.which(str(bliss_executable))
    if executable is None:
        raise FileNotFoundError(f'BLISS executable not found: {bliss_executable}; set --bliss-executable')
    executable = str(Path(executable).resolve())
    outdir.mkdir(parents=True, exist_ok=True)
    stem = h5_path.stem
    dat_path = outdir / f'{stem}_bliss_hits.dat'
    csv_path = outdir / f'{stem}_bliss_hits.csv'
    manifest_path = outdir / f'{stem}_bliss_search.json'
    unity_path = outdir / f'{stem}_unity.f32'
    log_path = outdir / f'{stem}_bliss.log'
    identity = {'schema': 1, 'input': file_identity(h5_path), 'executable': file_identity(executable),
                'layout': layout, 'device': device, 'snr': snr, 'min_drift': min_drift,
                'max_drift': max_drift, 'drift_step': drift_step, 'distance': distance,
                'equalizer': 'unity'}
    cached = False
    if not force and manifest_path.exists() and dat_path.exists():
        try:
            previous = json.loads(manifest_path.read_text())
            cached = previous['identity'] == identity and previous['dat_sha256'] == sha256(dat_path)
        except (ValueError, KeyError):
            pass
    with tempfile.TemporaryDirectory(prefix=f'.{stem}_bliss_', dir=outdir) as staging:
        staging = Path(staging)
        result = dat_path
        if not cached:
            unity = staging / 'unity.f32'
            np.ones(layout['fine_channels_per_coarse'], dtype=np.float32).tofile(unity)
            result = staging / 'hits.dat'
            command = [executable, str(h5_path), '--nchan-per-coarse', str(layout['fine_channels_per_coarse']),
                       '--coarse-channel', str(layout['first_coarse']), '--number-coarse', str(layout['number_coarse']),
                       '-e', str(unity), '--distance', str(distance), '-s', str(snr),
                       '-md', str(min_drift), '-MD', str(max_drift), '-rs', str(drift_step)]
            if device is not None:
                command += ['--device', device]
            command += ['-o', str(result), 'dat']
            print('Running BLISS directly:', ' '.join(command), flush=True)
            with open(log_path, 'w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not result.exists():
                raise RuntimeError(f'BLISS returned without producing a .dat file; see {log_path}')
            if file_identity(h5_path) != identity['input']:
                raise RuntimeError('Input HDF5 changed during the search; results were not published')
        hits = read_dat(result)
        cc = hits['Coarse_Channel_Number']
        if ((cc < layout['first_coarse']) | (cc >= layout['first_coarse'] + layout['number_coarse'])).any():
            raise ValueError('BLISS reported hits outside the requested coarse-channel interval')
        if (hits['Index'] >= layout['fine_channels_per_coarse']).any():
            raise ValueError('BLISS returned an out-of-range coarse-local fine-channel index')
        pending_csv = staging / 'hits.csv'
        hits.to_csv(pending_csv, index=False)
        manifest = {'identity': identity, 'dat_sha256': sha256(result), 'hit_count': len(hits)}
        pending_manifest = staging / 'manifest.json'
        pending_manifest.write_text(json.dumps(manifest, indent=2))
        if not cached:
            os.replace(result, dat_path)
            os.replace(unity, unity_path)
        os.replace(pending_csv, csv_path)
        os.replace(pending_manifest, manifest_path)
    print(f'{"Reused verified .dat; regenerated" if cached else "Wrote"} {len(hits)} hits: {csv_path}')
    return str(csv_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--h5', required=True)
    parser.add_argument('--outdir', default='.')
    parser.add_argument('--force', action='store_true')
    add_search_arguments(parser)
    args = parser.parse_args()
    run_search(args.h5, args.outdir, args.bliss_executable, args.device, args.snr,
               args.min_drift, args.max_drift, args.drift_step, args.distance, args.force)
