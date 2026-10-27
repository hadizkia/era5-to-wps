#!/usr/bin/env python3
"""Self-tests that need no real ERA5:

  1. write_field / read_file round trip of the WPS intermediate format
  2. the lat-lon window, including one that crosses 0/360 and the whole globe
  3. the converter end to end on a tiny synthetic ds633.0 tree, read back

    python3 tests/test_era5_to_wps.py
"""
import datetime as dt
import os
import subprocess
import sys
import tempfile

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
TOP = os.path.dirname(HERE)
sys.path.insert(0, TOP)
from wps_intermediate import read_file, write_field   # noqa: E402
from era5_to_wps import Window, PL, SFC                # noqa: E402


def test_roundtrip():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'X:2000-01-01_00')
        with open(p, 'wb') as fh:
            write_field(fh, '2000-01-01_00:00:00', 'TT', 'K', 'Temperature', 85000., a,
                        startlat=10., startlon=200., dlat=0.25, dlon=0.25)
            write_field(fh, '2000-01-01_00:00:00', 'PMSL', 'Pa', 'Sea-level Pressure', 201300., a + 1,
                        startlat=10., startlon=200., dlat=0.25, dlon=0.25)
        f = read_file(p)
    assert [x['field'] for x in f] == ['TT', 'PMSL']
    assert f[0]['xlvl'] == 85000. and f[0]['iproj'] == 0 and f[0]['startloc'] == 'SWCORNER'
    assert (f[0]['startlat'], f[0]['startlon'], f[0]['dlat'], f[0]['dlon']) == (10., 200., 0.25, 0.25)
    assert np.array_equal(f[0]['data'], a) and np.array_equal(f[1]['data'], a + 1)
    print('1. round trip: OK')


def test_window():
    lat = np.arange(90, -90.01, -0.25)       # ERA5: north -> south
    lon = np.arange(0, 360, 0.25)
    w = Window(lat, lon, -40, 85, 110, 359.75)
    assert w.startlat == -40 and w.dlat == 0.25 and w.startlon == 110 and w.dlon == 0.25
    assert w.j.size == 501 and w.i.size == 1000
    w = Window(lat, lon, 20, 50, 350, 10)    # crosses 0/360
    assert w.startlon == 350 and w.dlon == 0.25 and w.i.size == 81
    assert lon[w.i[39]] == 359.75 and lon[w.i[40]] == 0
    w = Window(lat, lon, 20, 50, -130, -60)  # negative longitudes
    assert w.startlon == 230 and w.i.size == 281
    w = Window(lat, lon, -90, 90, 0, 360)    # whole globe
    assert w.i.size == 1440 and w.j.size == 721
    a = np.random.rand(3, lat.size, lon.size)
    w = Window(lat, lon, 20, 50, 350, 10)
    c = w.cut(a)
    assert c.shape == (3, 121, 81)
    assert np.allclose(c[:, 0, 0], a[:, np.where(lat == 20)[0][0], np.where(lon == 350)[0][0]])
    print('2. window: OK')


def make_fake_rda(top, t, lat, lon, levels):
    """minimal ds633.0 layout with constant fields, one day / one month"""
    d = t.strftime('%Y%m%d'); ym = t.strftime('%Y%m')
    times = [np.datetime64(t + dt.timedelta(hours=h)) for h in range(24)]
    os.makedirs(f'{top}/e5.oper.an.pl/{ym}'); os.makedirs(f'{top}/e5.oper.an.sfc/{ym}')
    os.makedirs(f'{top}/e5.oper.invariant/197901')
    vals = {'Z': 9810., 'T': 250., 'U': 5., 'V': -3., 'Q': 2e-3}
    for name, (code, var, *_x) in PL.items():
        a = np.full((24, levels.size, lat.size, lon.size), vals[var], np.float32)
        xr.Dataset({var: (('time', 'level', 'latitude', 'longitude'), a)},
                   coords=dict(time=times, level=levels, latitude=lat, longitude=lon)
                   ).to_netcdf(f'{top}/e5.oper.an.pl/{ym}/e5.oper.an.pl.{code}.ll025sc.{d}00_{d}23.nc')
    sv = {'2t': 290., '2d': 280., '10u': 1., '10v': 2., 'sp': 1e5, 'msl': 101000., 'skt': 291.,
          'sstk': 292., 'ci': 0., 'sd': 0.01, 'rsn': 200., 'stl1': 285., 'stl2': 284., 'stl3': 283.,
          'stl4': 282., 'swvl1': .3, 'swvl2': .29, 'swvl3': .28, 'swvl4': .27}
    for k, code in SFC.items():
        a = np.full((24, lat.size, lon.size), sv[k], np.float32)
        if k == 'sstk':
            a[:, 0, 0] = np.nan                       # "land" point: missing SST
        v = code.split('_')[-1].upper()
        xr.Dataset({v: (('time', 'latitude', 'longitude'), a)},
                   coords=dict(time=times, latitude=lat, longitude=lon)
                   ).to_netcdf(f'{top}/e5.oper.an.sfc/{ym}/e5.oper.an.sfc.{code}.ll025sc.{ym}0100_{ym}3123.nc')
    for code, v, val in (('128_172_lsm', 'LSM', 1.), ('128_129_z', 'Z', 981.)):
        a = np.full((1, lat.size, lon.size), val, np.float32)
        xr.Dataset({v: (('time', 'latitude', 'longitude'), a)},
                   coords=dict(time=[np.datetime64('1979-01-01')], latitude=lat, longitude=lon)
                   ).to_netcdf(f'{top}/e5.oper.invariant/197901/e5.oper.invariant.{code}.ll025sc.1979010100_1979010100.nc')


def test_end_to_end():
    t = dt.datetime(2000, 1, 1)
    lat = np.arange(50, 19.9, -0.25); lon = np.arange(250, 270.01, 0.25)
    levels = np.array([1000., 850., 500.])
    with tempfile.TemporaryDirectory() as d:
        make_fake_rda(f'{d}/rda', t, lat, lon, levels)
        subprocess.run([sys.executable, f'{TOP}/era5_to_wps.py', '--rda', f'{d}/rda', '--start', '2000-01-01_00',
                        '--end', '2000-01-01_03', '--hours', '3', '--out', f'{d}/out',
                        '--window', '25', '45', '255', '265'], check=True)
        files = sorted(os.listdir(f'{d}/out'))
        assert files == ['ERA5:2000-01-01_00', 'ERA5:2000-01-01_03'], files
        f = read_file(f'{d}/out/{files[0]}')
    by = {(x['field'], x['xlvl']): x for x in f}
    assert len(f) == 6 * levels.size + 20 + 1                     # 6 upper-air fields, 20 surface, PMSL
    hgt = by[('HGT', 85000.)]
    assert hgt['data'].shape == (81, 41) and hgt['startlat'] == 25 and hgt['startlon'] == 255
    assert np.allclose(hgt['data'], 9810. / 9.81)
    assert np.allclose(by[('SPECHUMD', 50000.)]['data'], 2e-3)
    assert np.allclose(by[('SNOW', 200100.)]['data'], 10.) and np.allclose(by[('SNOWH', 200100.)]['data'], 0.05)
    assert np.allclose(by[('SOILHGT', 200100.)]['data'], 100.)
    rh = by[('RH', 200100.)]['data']; assert 0 < rh.min() and rh.max() <= 100
    assert np.allclose(by[('PMSL', 201300.)]['data'], 101000.)
    print('3. end to end on synthetic ds633.0: OK (%d records per time)' % len(f))


if __name__ == '__main__':
    test_roundtrip()
    test_window()
    test_end_to_end()
    print('all tests passed')
