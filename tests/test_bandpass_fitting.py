"""Bandpass recovery, data preservation, and interrupted-write tests."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np

from first_half_pipeline.upchannelize import coarse_layout
from second_half_pipeline import lwa_bliss_bp_gen as bp
from second_half_pipeline import chunk_and_bliss


class BandpassTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'synthetic.h5'
        self.export = Path(self.tmp.name) / 'synthetic_bpmodel.f32'
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(mock.patch.object(bp, 'array_backend', return_value=np))

    def fixture(self, nchans=512, coarse=8, edges=1, descending=False, ripple=False):
        layout = coarse_layout(nchans, coarse, edges)
        start, stop = layout['valid_channel_start'], layout['valid_channel_stop']
        x = np.linspace(-1, 1, stop - start)
        shape = 1 + 0.2 * x + 0.1 * x ** 2
        if ripple:
            shape += 0.03 * (-1.0) ** np.arange(len(x))
        scale = np.arange(7) * 0.1 + 0.7  # median = 1, allowing known recovery
        raw = np.full((7, 1, nchans), -1.0, dtype='float32')
        raw[:, 0, start:stop] = 100 * scale[:, None] * shape[None, :]
        mask = np.ones(raw.shape, dtype='uint8')
        mask[:, :, start:stop] = 0
        with h5py.File(self.path, 'w') as f:
            f.attrs.update(CLASS='FILTERBANK', VERSION='1.0')
            ds = f.create_dataset('data', data=raw)
            ds.attrs.update(layout)
            ds.attrs.update(nchans=nchans, fch1=60.0, foff=-0.001 if descending else 0.001,
                            nbits=32, tstart=60000.0, tsamp=3.0, source_name='synthetic',
                            src_raj=12.5, src_dej=-30.0)
            for axis, label in enumerate(('time', 'feed_id', 'frequency')):
                ds.dims[axis].label = label
            f.create_dataset('mask', data=mask)
            f['mask'].attrs['description'] = 'test mask'
        return raw, mask, shape, start, stop

    def fit(self, **kwargs):
        return bp.computeBandpassData_streaming(self.path, chunk_size=3,
                                                channel_block=17, **kwargs)

    def test_flattening_layouts_and_metadata_preservation(self):
        for n, coarse, edges, descending in ((512, 8, 1, False), (64, 32, 3, True),
                                             (16, 1, 0, False), (3, 3, 1, False)):
            with self.subTest(nchans=n, coarse=coarse, edges=edges):
                raw, mask, shape, start, stop = self.fixture(n, coarse, edges, descending)
                profile = self.fit()
                with h5py.File(self.path) as f:
                    self.assertEqual(set(f), {'data', 'uncorrected', 'mask', 'bandpass_model'})
                    np.testing.assert_array_equal(f['uncorrected'][:], raw)
                    np.testing.assert_array_equal(f['mask'][:], mask)
                    np.testing.assert_array_equal(f['bandpass_model'][:], profile)
                    np.testing.assert_array_equal(profile[:start], 1.0)
                    np.testing.assert_array_equal(profile[stop:], 1.0)
                    self.assertTrue(np.all(np.isfinite(profile)) and np.all(profile > 0))
                    np.testing.assert_allclose(profile[start:stop], shape / shape.mean(), rtol=1e-5)
                    expected = raw.copy()
                    expected[:, 0, start:stop] = (raw[:, 0, start:stop].astype('float64') /
                                                  profile[start:stop]).astype('float32')
                    np.testing.assert_array_equal(f['data'][:], expected)
                    flattened = f['data'][:, 0, start:stop].astype('float64')
                    # Float32 output and high-order SG fitting introduce rounding;
                    # require sub-ppm residual variation for this smooth fixture.
                    self.assertLess(np.max(np.std(flattened, axis=1) /
                                           np.mean(flattened, axis=1)), 1e-6)
                    for key, value in f['uncorrected'].attrs.items():
                        np.testing.assert_equal(f['data'].attrs[key], value)
                    self.assertEqual([dim.label for dim in f['data'].dims], ['time', 'feed_id', 'frequency'])
                    self.assertEqual(f.attrs['CLASS'], 'FILTERBANK')
                    self.assertEqual(f['mask'].attrs['description'], 'test mask')
                    self.assertEqual(f['data'].attrs['source_dataset'], 'uncorrected')

    def test_final_smoothing_is_applied_and_exported_exactly(self):
        raw, _, shape, start, stop = self.fixture(ripple=True)
        with mock.patch.object(bp, 'compute_bandpass_instrumental', return_value=np.ones(stop - start)) as instrument:
            bp.main(self.path, self.export, window_size=1, final_window=21, channel_block=17)
        self.assertEqual(instrument.call_count, 1)
        with h5py.File(self.path) as f:
            profile = f['bandpass_model'][:]
            np.testing.assert_array_equal(np.fromfile(self.export, dtype='float32'), profile)
            self.assertGreater(np.max(np.abs(profile[start:stop] - shape / shape.mean())), 0.01)
            expected = (raw[:, 0, start:stop].astype('float64') / profile[start:stop]).astype('float32')
            np.testing.assert_array_equal(f['data'][:, 0, start:stop], expected)
            self.assertEqual(f['bandpass_model'].attrs['final_savgol_window'], 21)
        # A rerun recreates the export without recomputing hardware response.
        self.export.unlink()
        with mock.patch.object(bp, 'compute_bandpass_instrumental', side_effect=AssertionError('unexpected refit')):
            bp.main(self.path, self.export)
        np.testing.assert_array_equal(np.fromfile(self.export, dtype='float32'), profile)

    def test_instrument_ignores_excluded_values_and_uses_valid_frequencies(self):
        _, _, _, start, stop = self.fixture(descending=True)
        instrument = np.full(512, np.nan)
        instrument[start:stop] = 1
        profile = self.fit(instr_bandpass=instrument)
        np.testing.assert_array_equal(self.fit(force=True), profile)
        with mock.patch.object(bp, 'compute_bandpass_instrumental', return_value=np.ones(stop - start)) as hardware:
            bp.main(self.path, self.export, force=True)
        np.testing.assert_allclose(hardware.call_args.args[0],
                                   (60 - 0.001 * np.arange(start, stop)) * 1e6)

    def test_force_refits_original_without_double_correction(self):
        raw, _, _, _, _ = self.fixture(ripple=True)
        first = self.fit(final_window=9)
        with h5py.File(self.path) as f:
            first_data = f['data'][:]
        np.testing.assert_array_equal(self.fit(final_window=31), first)  # default skips
        second = self.fit(force=True, final_window=31)
        self.assertGreater(np.max(np.abs(first - second)), 1e-5)
        with h5py.File(self.path) as f:
            np.testing.assert_array_equal(f['uncorrected'][:], raw)
            expected = raw.copy()
            start, stop = bp.read_layout(f['data'], f['mask'])
            expected[:, 0, start:stop] = (raw[:, 0, start:stop].astype('float64') /
                                         second[start:stop]).astype('float32')
            np.testing.assert_array_equal(f['data'][:], expected)
            self.assertFalse(np.array_equal(first_data, expected))

    def test_invalid_metadata_and_profiles_leave_input_untouched(self):
        for mutation in ('missing', 'bounds', 'nchans', 'instrument'):
            with self.subTest(mutation=mutation):
                raw, _, _, start, stop = self.fixture()
                with h5py.File(self.path, 'r+') as f:
                    if mutation == 'missing':
                        del f['data'].attrs['num_coarse']
                    elif mutation == 'bounds':
                        f['data'].attrs['valid_channel_stop'] = stop - 1
                    elif mutation == 'nchans':
                        f['data'].attrs['nchans'] = 511
                args = {'instr_bandpass': np.zeros(512)} if mutation == 'instrument' else {}
                with self.assertRaises(ValueError):
                    self.fit(**args)
                with h5py.File(self.path) as f:
                    self.assertEqual(set(f), {'data', 'mask'})
                    np.testing.assert_array_equal(f['data'][:], raw)

    def test_failed_write_preserves_raw_and_force_restarts(self):
        raw, _, _, start, stop = self.fixture()
        with h5py.File(self.path, 'r+') as f:
            f['data'][3, 0, start + 1] = np.nan  # not among the two sampled rows
        with self.assertRaisesRegex(ValueError, 'Nonfinite data'):
            self.fit(bpm_estimation_chunks=2)
        with h5py.File(self.path, 'r+') as f:
            self.assertNotIn('uncorrected', f)
            self.assertEqual(f[bp.WORK_GROUP].attrs['state'], 'writing')
            f['data'][3, 0, start + 1] = raw[3, 0, start + 1]
        with self.assertRaisesRegex(ValueError, '--force'):
            self.fit()
        self.fit(force=True)
        with h5py.File(self.path) as f:
            self.assertNotIn(bp.WORK_GROUP, f)
            np.testing.assert_array_equal(f['uncorrected'][:], raw)

    def test_ready_promotion_recovers_missing_public_data(self):
        raw, _, _, _, _ = self.fixture()
        with mock.patch.object(bp, '_promote', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.fit()
        with h5py.File(self.path, 'r+') as f:
            self.assertEqual(f[bp.WORK_GROUP].attrs['state'], 'ready')
            f['uncorrected'] = f['data']
            del f['data']  # simulate interruption between the two public-link updates
        with mock.patch.object(bp, 'smooth_bpmodel', side_effect=AssertionError('unexpected refit')):
            self.fit()
        with h5py.File(self.path) as f:
            np.testing.assert_array_equal(f['uncorrected'][:], raw)
            self.assertNotIn(bp.WORK_GROUP, f)
            self.assertEqual(f['data'].attrs['bandpass_correction_version'], 1)

    def test_failed_forced_refit_retains_previous_committed_result(self):
        raw, _, _, _, _ = self.fixture()
        profile = self.fit()
        with h5py.File(self.path) as f:
            corrected = f['data'][:]
        with mock.patch.object(bp, '_correct_blocks', side_effect=RuntimeError('interrupted')):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                self.fit(force=True)
        with h5py.File(self.path) as f:
            np.testing.assert_array_equal(f['data'][:], corrected)
            np.testing.assert_array_equal(f['bandpass_model'][:], profile)
            np.testing.assert_array_equal(f['uncorrected'][:], raw)
        with self.assertRaisesRegex(ValueError, '--force'):
            self.fit()
        np.testing.assert_array_equal(self.fit(force=True), profile)

    def test_bad_mask_or_sentinel_is_not_promoted(self):
        for bad_mask in (True, False):
            self.fixture()
            with h5py.File(self.path, 'r+') as f:
                f['mask' if bad_mask else 'data'][0, 0, 0] = 0
            with self.assertRaisesRegex(ValueError, 'Mask|edge samples'):
                self.fit()
            with h5py.File(self.path) as f:
                self.assertNotIn('uncorrected', f)

    def test_export_cannot_overwrite_hdf5(self):
        raw, _, _, _, _ = self.fixture()
        with self.assertRaisesRegex(ValueError, 'overwrite'):
            bp.main(self.path, self.path)
        with h5py.File(self.path) as f:
            np.testing.assert_array_equal(f['data'][:], raw)
            self.assertEqual(set(f), {'data', 'mask'})

    def test_legacy_chunker_rejects_double_correction(self):
        self.fixture()
        self.fit()
        outdir = Path(self.tmp.name) / 'hits'
        with mock.patch.object(chunk_and_bliss, 'run_bliss_on_chunk') as bliss:
            with self.assertRaisesRegex(ValueError, 'bandpass twice'):
                chunk_and_bliss.process_file(self.path, self.export, outdir, 512, 64, 8)
        bliss.assert_not_called()
        self.assertFalse(outdir.exists())


if __name__ == '__main__':
    unittest.main()
