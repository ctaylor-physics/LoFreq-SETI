"""Exercise direct-search orchestration with a fake external BLISS executable."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from first_half_pipeline.upchannelize import coarse_layout
from second_half_pipeline.hits_io import HIT_COLUMNS, read_dat, read_hits_csv
from second_half_pipeline.run_bliss import run_search, search_layout

ROOT = Path(__file__).resolve().parents[1]
HEADER = '# Top_Hit_# Drift_Rate SNR Uncorrected_Frequency Corrected_Frequency Index freq_start freq_end SEFD_freq Coarse_Channel_Number Full_number_of_hits\n'
ROW = '1 -0.135633 18.554905 56.000020 56.000020 4 56.000019 56.000021 0 0 1 4\n'


def fixture(path, channels=128, coarse=8, edge=1):
    with h5py.File(path, 'w') as f:
        ds = f.create_dataset('data', data=np.ones((4, 1, channels), dtype='f4'))
        ds.attrs.update(coarse_layout(channels, coarse, edge))
        ds.attrs.update(nchans=channels, fch1=56., foff=1e-6, tsamp=1.,
                        bandpass_correction_version=1, edge_mask_value=-1.)
        f.create_dataset('uncorrected', data=ds[:])
        f.create_dataset('mask', data=np.zeros(ds.shape, dtype='u1'))
        model = f.create_dataset('bandpass_model', data=np.ones(channels, dtype='f4'))
        model.attrs['bandpass_correction_version'] = 1


class DirectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='bliss test ')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.h5 = self.root / 'sample.h5'
        fixture(self.h5)
        self.exe = self.root / 'fake bliss'
        self.count = self.root / 'calls'
        self.exe.write_text(f'''#!{sys.executable}
import sys, pathlib, array
args = sys.argv[1:]
def opt(key): return args[args.index(key)+1]
assert args[-1] == 'dat'
width = int(opt('--nchan-per-coarse'))
a = array.array('f')
with open(opt('-e'), 'rb') as f: a.fromfile(f, width)
assert len(a) == width and all(x == 1 for x in a)
assert opt('--coarse-channel') == '1' and opt('--number-coarse') == '6'
p = pathlib.Path({str(self.count)!r})
p.write_text(p.read_text() + '1' if p.exists() else '1')
pathlib.Path(opt('-o')).write_text({(HEADER + ROW)!r})
''')
        self.exe.chmod(0o755)
        self.out = self.root / 'results'

    def test_parser_schema_and_empty(self):
        dat = self.root / 'hits.dat'
        dat.write_text(HEADER + ROW)
        row = read_dat(dat).iloc[0]
        self.assertEqual(row['SNR'], 18.554905)
        self.assertEqual(row['SEFD'], 0)
        self.assertEqual(row['Coarse_Channel_Number'], 1)
        self.assertEqual(row['Bin_Width'], 4)
        dat.write_text(HEADER)
        self.assertEqual(list(read_dat(dat).columns), HIT_COLUMNS)
        self.assertEqual(len(read_dat(dat)), 0)
        for bad in [ROW.rsplit(' ', 1)[0], ROW.replace('18.554905', 'nan'), ROW + 'extra']:
            dat.write_text(HEADER + bad)
            with self.assertRaises(ValueError):
                read_dat(dat)
        csv = self.root / 'legacy.csv'
        csv.write_text('Top_Hit_#,SNR\n1,10\n')
        with self.assertRaisesRegex(ValueError, 'regenerate'):
            read_hits_csv(csv)

    def test_layout_varies_and_rejects_old(self):
        fixture(self.h5, channels=256, coarse=16, edge=2)
        self.assertEqual(search_layout(self.h5), dict(fine_channels_per_coarse=16, first_coarse=2, number_coarse=12))
        with h5py.File(self.h5, 'r+') as f:
            del f['data'].attrs['bandpass_correction_version']
        with self.assertRaises(ValueError):
            search_layout(self.h5)

    def test_search_cache_and_invalidation(self):
        csv = Path(run_search(self.h5, self.out, self.exe, device='cpu'))
        self.assertEqual(len(read_hits_csv(csv)), 1)
        self.assertEqual(list(self.out.glob('*.h5')), [])
        np.testing.assert_array_equal(np.fromfile(self.out/'sample_unity.f32', dtype='f4'), np.ones(16))
        csv.write_text('stale CSV')
        run_search(self.h5, self.out, self.exe, device='cpu')
        self.assertEqual(self.count.read_text(), '1')
        self.assertEqual(len(read_hits_csv(csv)), 1)
        run_search(self.h5, self.out, self.exe, device='cpu', snr=12)
        self.assertEqual(self.count.read_text(), '11')
        with h5py.File(self.h5, 'r+') as f:
            f['data'][0, 0, 20] = 2
        run_search(self.h5, self.out, self.exe, device='cpu', snr=12)
        self.assertEqual(self.count.read_text(), '111')

    def test_failure_does_not_publish(self):
        self.exe.write_text(f'#!{sys.executable}\nraise SystemExit(3)\n')
        with self.assertRaises(subprocess.CalledProcessError):
            run_search(self.h5, self.out, self.exe)
        self.assertFalse(list(self.out.glob('*.json')))
        self.assertFalse(list(self.out.glob('*.dat')))

    def test_missing_malformed_and_out_of_range_output(self):
        for payload in (None, HEADER + 'broken row\n', HEADER + ROW.replace('0 0 1 4', '0 0 0 4')):
            self.exe.write_text(f'#!{sys.executable}\nimport sys, pathlib\n' +
                ('' if payload is None else
                 f'pathlib.Path(sys.argv[sys.argv.index("-o")+1]).write_text({payload!r})\n'))
            with self.assertRaises((RuntimeError, ValueError)):
                run_search(self.h5, self.out, self.exe)
            self.assertFalse(list(self.out.glob('*.json')))
            self.assertFalse(list(self.out.glob('*.dat')))

    def test_bandpass_reuse_and_force(self):
        from unittest import mock
        from second_half_pipeline import run_pipeline as pipeline
        self.out.mkdir()
        before = self.h5.stat().st_mtime_ns
        with mock.patch.object(pipeline, 'run') as run:
            export = pipeline.make_bandpass(str(self.h5), str(self.out), str(ROOT/'second_half_pipeline'), False)
            run.assert_not_called()
            self.assertEqual(self.h5.stat().st_mtime_ns, before)
            np.testing.assert_array_equal(np.fromfile(export, dtype='f4'), np.ones(128))
            with self.assertRaisesRegex(ValueError, '--force'):
                pipeline.make_bandpass(str(self.h5), str(self.out), '', False, fit_clip=True)
            pipeline.make_bandpass(str(self.h5), str(self.out), '', True, fit_clip=True)
            self.assertIn('--force', run.call_args.args[0])
            self.assertIn('--fit-clip', run.call_args.args[0])
            with h5py.File(self.h5, 'r+') as f:
                del f['data'].attrs['bandpass_correction_version']
            run.reset_mock()
            pipeline.make_bandpass(str(self.h5), str(self.out), '', False)
            run.assert_called_once()  # A stale .f32 must not bypass correction.

    def test_full_pair_pipeline_from_other_directory(self):
        inputs = []
        for station in ('LWA1', 'LWA-SV'):
            for tuning in (1, 2):
                path = self.root / f'observation-{station}_tun{tuning}.h5'
                fixture(path)
                inputs.append(str(path))
        command = [sys.executable, str(ROOT/'second_half_pipeline/run_pipeline.py'), *inputs,
                   '--bliss-executable', str(self.exe), '--outdir', str(self.out), '--width', '8']
        result = subprocess.run(command, cwd=self.root, env={**os.environ, 'MPLBACKEND': 'Agg'}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.count.read_text(), '1111')
        for tuning in (1, 2):
            self.assertEqual(len(read_hits_csv(self.out/f'tun{tuning}/hits_ONE_SV_ONE.csv')), 1)
            self.assertTrue((self.out/f'tun{tuning}/plots/pair_coincidence_stamps.pdf').exists())
        result = subprocess.run(command, cwd=self.root, env={**os.environ, 'MPLBACKEND': 'Agg'}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.count.read_text(), '1111')

    def test_pair_alignment_and_empty_triples(self):
        dat = self.root/'source.dat'
        dat.write_text(HEADER + ROW)
        a = read_dat(dat)
        one, sv, na = [self.root/f'{s}.csv' for s in ('one', 'sv', 'na')]
        a.to_csv(one, index=False)
        import pandas as pd
        pd.concat([a, a], ignore_index=True).to_csv(sv, index=False)
        command = [sys.executable, str(ROOT/'second_half_pipeline/anti_coincidence.py'), '--one', str(one), '--sv', str(sv), '--outdir', str(self.out)]
        subprocess.run(command, check=True, capture_output=True)
        self.assertEqual(len(read_hits_csv(self.out/'hits_ONE_SV_ONE.csv')), 2)
        self.assertEqual(len(read_hits_csv(self.out/'hits_ONE_SV_SV.csv')), 2)
        a.iloc[:0].to_csv(na, index=False)
        subprocess.run(command + ['--na', str(na)], check=True, capture_output=True)
        for station in ('ONE', 'NA', 'SV'):
            self.assertEqual(len(read_hits_csv(self.out/f'hits_all_three_{station}.csv')), 0)
