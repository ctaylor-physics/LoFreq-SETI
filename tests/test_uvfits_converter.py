"""Frequency, time, visibility, and flag semantics for UVFITS conversion."""
from pathlib import Path
import sys
import tempfile
import unittest

from astropy.io import fits
from blimpy import Waterfall
import h5py
import numpy as np

from uvfits_to_filterbank import convert, describe
from second_half_pipeline.hits_io import read_hits_csv
from second_half_pipeline.run_bliss import run_search


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'example.UVF'
        times = [0, 1, 2, 3, 100, 101, 102, 103]
        # Two baselines, two polarizations, two 3-IF windows, one missing weight.
        raw = np.zeros((16, 1, 1, 6, 1, 2, 3), dtype='f4')
        for i in range(16):
            raw[i, 0, 0, :, 0, 0, 0] = np.arange(6) + i
            raw[i, 0, 0, :, 0, 0, 1] = 4
            raw[i, 0, 0, :, 0, 1, 1] = 4
            raw[i, 0, 0, :, 0, :, 2] = 1
        raw[0, 0, 0, 1, 0, 0, 2] = 0
        date_floor = np.full(16, 2460000, dtype='f8')
        date_fraction = np.array([0.5 + t / 86400 for t in times for _ in (0, 1)])
        baseline = np.array([258, 259] * 8, dtype='f4')
        groups = fits.GroupData(raw, parnames=['DATE', 'DATE', 'BASELINE', 'INTTIM'],
                                pardata=[date_floor, date_fraction, baseline, np.ones(16)], bitpix=-32)
        primary = fits.GroupsHDU(groups)
        primary.header['CRVAL4'] = 100e6
        primary.header['CRPIX4'] = 1
        primary.header['CDELT4'] = 1e6
        primary.header['CRVAL3'] = -1
        primary.header['CDELT3'] = -1
        primary.header['OBJECT'] = 'TARGET'
        primary.header['OBSRA'] = 202.7845
        primary.header['OBSDEC'] = -30.5
        primary.header['BUNIT'] = 'UNCALIB'
        fq = fits.BinTableHDU.from_columns([
            fits.Column(name='IF FREQ', format='6D', array=[[0, 1e6, 2e6, 10e6, 11e6, 12e6]]),
            fits.Column(name='CH WIDTH', format='6D', array=[[1e6] * 6]),
            fits.Column(name='SIDEBAND', format='6J', array=[[1] * 6]),
        ], name='AIPS FQ')
        fits.HDUList([primary, fq]).writeto(self.source)

    def test_windows_scans_flags_and_filterbank_reader(self):
        with fits.open(self.source) as hdul:
            baselines, stokes, windows = describe(hdul)
        self.assertEqual(baselines, [258, 259])
        self.assertEqual(stokes, [-1, -2])
        self.assertEqual([(a, b) for a, b, _ in windows], [(0, 3), (3, 6)])
        paths = convert(self.source, self.root/'out', baselines='258')
        self.assertEqual(len(paths), 4)  # two time segments x two IF windows
        self.assertEqual(Waterfall(str(paths[0]), load_data=False).header['nchans'], 3)
        with h5py.File(paths[0]) as f:
            ds = f['data']
            self.assertEqual(ds.shape, (4, 1, 3))
            self.assertAlmostEqual(ds[0, 0, 0], 4)
            self.assertEqual(ds[0, 0, 1], 0)
            self.assertEqual(f['mask'][0, 0, 1], 1)
            self.assertEqual(ds.attrs['fch1'], 100)
            self.assertEqual(ds.attrs['foff'], 1)
            self.assertEqual(ds.attrs['uvfits_measure'], 'amplitude')
            self.assertEqual(ds.attrs['uvfits_stokes_code'], -1)
            self.assertAlmostEqual(ds.attrs['tstart'], 60000.0)
            self.assertAlmostEqual(ds.attrs['tsamp'], 1)
            self.assertLess(abs((f['time_mjd'][-1] - f['time_mjd'][0]) * 86400 - 3), 0.01)
            self.assertAlmostEqual(ds.attrs['src_raj'], 202.7845 / 15)
            self.assertAlmostEqual(ds.attrs['src_dej'], -30.5)
        with h5py.File(paths[1]) as f:
            self.assertEqual(f['data'].attrs['fch1'], 110)
        with h5py.File(paths[2]) as f:
            self.assertAlmostEqual(f['data'].attrs['tstart'], 60000.0 + 100 / 86400, places=6)
        with self.assertRaises(FileExistsError):
            convert(self.source, self.root/'out', baselines='258')

    def test_select_polarization_and_power(self):
        paths = convert(self.source, self.root/'power', baselines='259', polarizations='1', measure='power')
        self.assertEqual(len(paths), 4)
        with h5py.File(paths[0]) as f:
            self.assertEqual(f['data'].attrs['uvfits_polarization_index'], 1)
            self.assertEqual(f['data'][0, 0, 0], 16)
        with self.assertRaisesRegex(ValueError, 'polarizations'):
            convert(self.source, self.root/'bad', polarizations='3')

    def test_single_output_runs_direct_search(self):
        paths = convert(self.source, self.root/'single', baselines='258',
                        scan_selection='1', window_selection='1')
        self.assertEqual(len(paths), 1)
        fake = self.root/'fake_bliss'
        fake.write_text(f'''#!{sys.executable}
import pathlib, sys, array
args = sys.argv[1:]
def option(key): return args[args.index(key) + 1]
assert option('--nchan-per-coarse') == '3'
assert option('--coarse-channel') == '0'
assert option('--number-coarse') == '1'
profile = array.array('f')
with open(option('-e'), 'rb') as stream: profile.fromfile(stream, 3)
assert list(profile) == [1, 1, 1]
pathlib.Path(option('-o')).write_text('# Top_Hit_# Drift_Rate SNR Uncorrected_Frequency Corrected_Frequency Index freq_start freq_end SEFD_freq Coarse_Channel_Number Full_number_of_hits\\n')
''')
        fake.chmod(0o755)
        csv = run_search(paths[0], self.root/'hits', fake, device='cpu')
        self.assertEqual(len(read_hits_csv(csv)), 0)

    def test_reject_unsupported_frequency_axis(self):
        with fits.open(self.source) as hdul:
            hdul[0].header['NAXIS4'] = 2
            with self.assertRaisesRegex(ValueError, 'NAXIS4=1'):
                describe(hdul)
