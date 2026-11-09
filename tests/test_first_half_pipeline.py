"""Synthetic HDF5 checks; no raw DRX observations or BLISS installation needed."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from first_half_pipeline import pipeline, upchannelize


class FirstHalfTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, cwd)
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def waterfall(self, nchans, second_nchans=None, descending=False):
        path = Path(self.tmp.name) / 'observation-LWA1-waterfall.hdf5'
        with h5py.File(path, 'w') as f:
            obs = f.create_group('Observation1')
            obs.attrs.update(RA=12.5, Dec=-30.0, tInt=3.0, TargetName='synthetic')
            obs.create_dataset('time', data=[[1700000000, 0.25], [1700000003, 0.25]])
            for tuning, count in ((1, nchans), (2, second_nchans or nchans)):
                group = obs.create_group(f'Tuning{tuning}')
                group.create_dataset('I', data=(np.arange(2 * count, dtype=np.float32)
                                               .reshape(2, count) + tuning))
                step = -2.0 if descending else 2.0
                group.create_dataset('freq', data=40e6 + tuning * 1e6 + step * np.arange(count))
        return path

    def test_layout_defaults_and_invalid_configurations(self):
        layout = upchannelize.coarse_layout(8388608)
        self.assertEqual(layout['fine_channels_per_coarse'], 262144)
        self.assertEqual(layout['valid_channel_start'], 786432)
        self.assertEqual(layout['valid_channel_stop'], 7602176)
        for args in ((64, 0, 0), (64, -1, 0), (64, 8, -1),
                     (64, 8, 4), (64, 8, 5), (65, 8, 1), (1, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                upchannelize.coarse_layout(*args)

    def test_full_grid_and_both_tunings_across_layouts(self):
        # Include no edges, one remaining coarse channel, descending frequencies,
        # and enough channels to cross the writer's bounded read block size.
        for nchans, coarse, edges, descending in (
            (64, 32, 3, False), (128, 8, 1, True), (32, 1, 0, False),
            (48, 3, 1, False), (262176, 32, 0, False),
        ):
            with self.subTest(nchans=nchans, coarse=coarse, edges=edges):
                source = self.waterfall(nchans, descending=descending)
                upchannelize.convert_waterfall(source, coarse, edges)
                start = edges * (nchans // coarse)
                stop = nchans - start
                for tuning in (1, 2):
                    with h5py.File(source) as original, h5py.File(
                        f'observation-LWA1_tun{tuning}.h5'
                    ) as output:
                        raw = original[f'Observation1/Tuning{tuning}/I'][:]
                        freq = original[f'Observation1/Tuning{tuning}/freq'][:] / 1e6
                        data = output['data']
                        self.assertEqual(data.shape, (2, 1, nchans))
                        self.assertEqual(data.dtype, np.dtype('float32'))
                        expected = np.full_like(raw, -1.0)
                        expected[:, start:stop] = raw[:, start:stop]
                        np.testing.assert_array_equal(data[:, 0, :], expected)
                        expected_mask = np.ones(raw.shape, dtype='uint8')
                        expected_mask[:, start:stop] = 0
                        np.testing.assert_array_equal(output['mask'][:, 0, :], expected_mask)
                        self.assertEqual(output['mask'].dtype, np.dtype('uint8'))
                        self.assertEqual(data.attrs['nchans'], nchans)
                        self.assertEqual(data.attrs['fch1'], freq[0])
                        self.assertEqual(data.attrs['foff'], freq[1] - freq[0])
                        self.assertEqual(data.attrs['src_raj'], 12.5)
                        self.assertEqual(data.attrs['src_dej'], -30.0)
                        self.assertEqual(data.attrs['tsamp'], 3.0)
                        self.assertEqual(data.attrs['source_name'], 'synthetic')
                        self.assertAlmostEqual(data.attrs['tstart'],
                                               1700000000.25 / 86400 + 40587, places=8)
                        self.assertEqual(output.attrs['CLASS'], 'FILTERBANK')
                        for key, value in upchannelize.coarse_layout(nchans, coarse, edges).items():
                            self.assertEqual(data.attrs[key], value)
                        for name in ('data', 'mask'):
                            self.assertEqual([dim.label for dim in output[name].dims],
                                             ['time', 'feed_id', 'frequency'])

    def test_invalid_second_tuning_creates_no_outputs(self):
        source = self.waterfall(64, second_nchans=65)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            upchannelize.convert_waterfall(source, 8, 1)
        self.assertEqual(list(Path('.').glob('*_tun*.h5')), [])
        self.assertTrue(source.exists())

    def test_pipeline_passes_layout_and_station(self):
        argv = ['pipeline', '--tar-path', 'input.tgz', '--drx-path', '/raw',
                '--avg', '3', '--length', '4096', '--station', 'lwa1',
                '--num-coarse', '8', '--edge-coarse', '1', '--meta-dir', 'metadata']
        with mock.patch.object(sys, 'argv', argv), \
             mock.patch('first_half_pipeline.get_metadata.build_csv') as metadata, \
             mock.patch.object(pipeline, 'run_upchannelize') as run:
            pipeline.main()
        metadata.assert_called_once_with('input.tgz', output_dir='metadata', station='lwa1')
        run.assert_called_once_with(tar_path='input.tgz', drx_path='/raw', avg=3.0,
                                    length=4096, meta_dir='metadata', num_coarse=8, edge_coarse=1)

    def test_all_stations_receive_layout(self):
        for station, label in (('lwa1', 'LWA1'), ('lwa-sv', 'LWA-SV'), ('lwa-na', 'LWA-NA')):
            Path(f'{station}.tgz').touch()
            Path(f'metadata_{station}.csv').write_text(
                f'tarball,beamid,datafile\n{station}.tgz,1,{station}.drx\n')
        with mock.patch.object(upchannelize, 'process_tarball') as process:
            upchannelize.run('*.tgz', '/raw', 3.0, 4096, num_coarse=8, edge_coarse=1)
        self.assertEqual(process.call_count, 3)
        self.assertEqual({call.args[5] for call in process.call_args_list}, {'LWA1', 'LWA-SV', 'LWA-NA'})
        for call in process.call_args_list:
            self.assertEqual(call.kwargs, {'num_coarse': 8, 'edge_coarse': 1})

    def test_cli_from_outside_repository_and_early_validation(self):
        repo = Path(upchannelize.__file__).resolve().parent.parent
        env = dict(os.environ, PYTHONPATH=str(repo), PYTHONDONTWRITEBYTECODE='1')
        for module in ('pipeline', 'upchannelize'):
            for invocation in ([sys.executable, '-m', f'first_half_pipeline.{module}'],
                               [sys.executable, str(repo / 'first_half_pipeline' / f'{module}.py')]):
                with self.subTest(invocation=invocation):
                    result = subprocess.run(invocation + ['--help'], env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--edge-coarse', result.stdout)
                    result = subprocess.run(invocation + [
                        '--tar-path', 'absent.tgz', '--drx-path', 'absent', '--avg', '3',
                        '--length', '65', '--num-coarse', '8', '--edge-coarse', '1',
                        '--meta-dir', 'must-not-be-created',
                    ], env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn('divisible', result.stderr)
                    self.assertFalse(Path('must-not-be-created').exists())


if __name__ == '__main__':
    unittest.main()
