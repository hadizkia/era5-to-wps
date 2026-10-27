#!/usr/bin/env python3
"""ERA5 (NCAR RDA ds633.0 NetCDF) -> WPS intermediate files for metgrid, replacing ungrib.

The fields are those ungrib produces from ERA5 with the ECMWF pressure-level
Vtable, after ungrib's rrpr derivations, plus specific humidity:

  pressure levels (37): HGT (= z / 9.81), TT, UU, VV, SPECHUMD, and RH from
                        SPECHUMD with ungrib's formula (compute_rh_spechumd_upa)
  surface (200100):     TT (2t), UU (10u), VV (10v), RH (2 m, from 2d),
                        PSFC, SKINTEMP, SST, SEAICE, LANDSEA, SOILHGT (= z_sfc / 9.81),
                        SNOW (= sd * 1000), SNOWH (= sd * 1000 / rsn),
                        ST000007 ST007028 ST028100 ST100289,
                        SM000007 SM007028 SM028100 SM100289
  mean sea level (201300): PMSL

Only a lat-lon window around the WRF domain is written, to keep the files
small (default 40S-85N, 110E-360E; the window may cross 0/360).

  python3 era5_to_wps.py --start 1979-01-01_00 --end 1979-02-01_00 --hours 3 --out DIR

Input: the ds633.0 directory tree (e5.oper.an.pl, e5.oper.an.sfc,
e5.oper.invariant), on NCAR GLADE by default or a local copy via --rda / $RDA.
"""
import argparse
import datetime as dt
import glob
import os

import numpy as np
import xarray as xr

from wps_intermediate import write_field

RDA_DEFAULT = os.environ.get("RDA", "/glade/campaign/collections/rda/data/d633000")
G_RRPR = 9.81        # gravity ungrib's rrpr uses to turn geopotential into height
PL = {"HGT": ("128_129_z", "Z", "m", "Height"), "TT": ("128_130_t", "T", "K", "Temperature"),
      "UU": ("128_131_u", "U", "m s-1", "U"), "VV": ("128_132_v", "V", "m s-1", "V"),
      "SPECHUMD": ("128_133_q", "Q", "kg kg-1", "Specific Humidity")}
SFC = {"2t": "128_167_2t", "2d": "128_168_2d", "10u": "128_165_10u", "10v": "128_166_10v",
       "sp": "128_134_sp", "msl": "128_151_msl", "skt": "128_235_skt", "sstk": "128_034_sstk",
       "ci": "128_031_ci", "sd": "128_141_sd", "rsn": "128_033_rsn",
       "stl1": "128_139_stl1", "stl2": "128_170_stl2", "stl3": "128_183_stl3", "stl4": "128_236_stl4",
       "swvl1": "128_039_swvl1", "swvl2": "128_040_swvl2", "swvl3": "128_041_swvl3", "swvl4": "128_042_swvl4"}


def one(pattern):
    f = sorted(glob.glob(pattern))
    if not f:
        raise SystemExit(f"missing ERA5 file: {pattern}")
    return f[0]


class Window:
    """index window on the ERA5 grid, written south -> north, west -> east"""

    def __init__(self, lat, lon, lat0, lat1, lon0, lon1):
        self.j = np.where((lat >= lat0) & (lat <= lat1))[0][::-1]       # ERA5 runs north -> south
        if lon1 - lon0 >= 360.0 - 1e-6:                                    # whole globe
            self.i = np.arange(lon.size)
        else:
            lon0, lon1 = lon0 % 360.0, lon1 % 360.0
            if lon0 <= lon1:
                self.i = np.where((lon >= lon0) & (lon <= lon1))[0]
            else:                                                          # crosses 0 / 360
                self.i = np.concatenate([np.where(lon >= lon0)[0], np.where(lon <= lon1)[0]])
        if self.j.size < 2 or self.i.size < 2:
            raise SystemExit("the lat-lon window holds fewer than 2 x 2 ERA5 points")
        self.startlat = float(lat[self.j[0]])
        self.startlon = float(lon[self.i[0]])
        self.dlat = float(lat[self.j[1]] - lat[self.j[0]])
        self.dlon = float((lon[self.i[1]] - lon[self.i[0]]) % 360.0)

    @classmethod
    def from_rda(cls, rda, lat0, lat1, lon0, lon1):
        with xr.open_dataset(one(f"{rda}/e5.oper.invariant/197901/*128_172_lsm*.nc")) as ds:
            return cls(ds.latitude.values, ds.longitude.values, lat0, lat1, lon0, lon1)

    def cut(self, a):            # (..., lat, lon) -> (..., ny south->north, nx)
        return np.asarray(a)[..., self.j, :][..., self.i].astype(np.float32)


class Reader:
    """reads ERA5 fields, keeping the daily pressure-level and monthly surface files open"""

    def __init__(self, rda, win):
        self.rda = rda
        self.win = win
        self.cache = {}

    def _open(self, key, path):
        if key not in self.cache:
            if len(self.cache) > 40:
                for k in list(self.cache)[:20]:
                    self.cache.pop(k).close()
            self.cache[key] = xr.open_dataset(path)
        return self.cache[key]

    def pl(self, code, var, t):
        d = t.strftime("%Y%m%d"); ym = t.strftime("%Y%m")
        ds = self._open(("pl", code, d), one(f"{self.rda}/e5.oper.an.pl/{ym}/e5.oper.an.pl.{code}.*.{d}00_{d}23.nc"))
        a = ds[var].sel(time=np.datetime64(t)).values          # (level, lat, lon)
        return ds["level"].values, self.win.cut(a)

    def sfc(self, code, t):
        ym = t.strftime("%Y%m")
        ds = self._open(("sfc", code, ym), one(f"{self.rda}/e5.oper.an.sfc/{ym}/e5.oper.an.sfc.{code}.*.nc"))
        v = [n for n in ds.data_vars if ds[n].ndim == 3][0]
        return self.win.cut(ds[v].sel(time=np.datetime64(t)).values)

    def inv(self, code):
        ds = self._open(("inv", code), one(f"{self.rda}/e5.oper.invariant/197901/*{code}*.nc"))
        v = [n for n in ds.data_vars if ds[n].ndim >= 2][0]
        a = ds[v].values
        return self.win.cut(a[0] if a.ndim == 3 else a)


def rh_from_td(t, td):
    """2 m RH from temperature and dew point, as ungrib's rrpr (Magnus, over water), %"""
    es = lambda x: 6.112 * np.exp(17.67 * (x - 273.15) / (x - 29.65))
    return np.clip(100.0 * es(td) / es(t), 0.0, 100.0)


def write_time(rd, t, outdir, prefix):
    w = rd.win
    hdate = t.strftime("%Y-%m-%d_%H:00:00")
    path = os.path.join(outdir, f"{prefix}:{t.strftime('%Y-%m-%d_%H')}")
    tmp = path + ".tmp"
    kw = dict(startlat=w.startlat, startlon=w.startlon, dlat=w.dlat, dlon=w.dlon)
    with open(tmp, "wb") as fh:
        keep = {}
        for name, (code, var, units, desc) in PL.items():
            lev, a = rd.pl(code, var, t)
            if name == "HGT":
                a = a / G_RRPR
            keep[name] = a
            for k, p in enumerate(lev):
                write_field(fh, hdate, name, units, desc, float(p) * 100.0, a[k], **kw)
        # RH from specific humidity on the source grid, exactly as ungrib's rrpr
        # (compute_rh_spechumd_upa): no clipping, p in Pa
        for k, p in enumerate(lev):
            pa = float(p) * 100.0; tt = keep["TT"][k]
            rh = 0.263 * pa * keep["SPECHUMD"][k] / np.exp(17.67 * (tt - 273.16) / (tt - 29.65))
            write_field(fh, hdate, "RH", "%", "Relative Humidity", pa, rh, **kw)
        s = {k: rd.sfc(c, t) for k, c in SFC.items()}
        sfc = [("TT", s["2t"], "K", "Temperature"), ("UU", s["10u"], "m s-1", "U"),
               ("VV", s["10v"], "m s-1", "V"), ("RH", rh_from_td(s["2t"], s["2d"]), "%", "Relative Humidity"),
               ("PSFC", s["sp"], "Pa", "Surface Pressure"),
               ("SKINTEMP", s["skt"], "K", "Skin temperature"),
               ("SST", s["sstk"], "K", "Sea-Surface Temperature"),
               ("SEAICE", s["ci"], "fraction", "Sea-Ice Fraction"),
               ("LANDSEA", rd.inv("128_172_lsm"), "0/1 Flag", "Land/Sea flag"),
               ("SOILHGT", rd.inv("128_129_z") / G_RRPR, "m", "Terrain field of source analysis"),
               ("SNOW", s["sd"] * 1000.0, "kg m-2", "Water Equivalent of Accumulated Snow Depth"),
               ("SNOWH", s["sd"] * 1000.0 / np.maximum(s["rsn"], 1.0), "m", "Physical Snow Depth"),
               ("ST000007", s["stl1"], "K", "T of 0-7 cm ground layer"),
               ("ST007028", s["stl2"], "K", "T of 7-28 cm ground layer"),
               ("ST028100", s["stl3"], "K", "T of 28-100 cm ground layer"),
               ("ST100289", s["stl4"], "K", "T of 100-289 cm ground layer"),
               ("SM000007", s["swvl1"], "m3 m-3", "Soil moisture of 0-7 cm ground layer"),
               ("SM007028", s["swvl2"], "m3 m-3", "Soil moisture of 7-28 cm ground layer"),
               ("SM028100", s["swvl3"], "m3 m-3", "Soil moisture of 28-100 cm ground layer"),
               ("SM100289", s["swvl4"], "m3 m-3", "Soil moisture of 100-289 cm ground layer")]
        for name, a, units, desc in sfc:
            # missing values (SST and sea ice over land) as metgrid's -1.e30
            write_field(fh, hdate, name, units, desc, 200100.0, np.nan_to_num(a, nan=-1.e30), **kw)
        write_field(fh, hdate, "PMSL", "Pa", "Sea-level Pressure", 201300.0, s["msl"], **kw)
    os.replace(tmp, path)
    return path


def _work(args):
    chunk, rda, window, out, prefix = args
    rd = Reader(rda, Window.from_rda(rda, *window))
    for t in chunk:
        write_time(rd, t, out, prefix)
    return len(chunk)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first time, YYYY-MM-DD_HH")
    ap.add_argument("--end", required=True, help="last time, YYYY-MM-DD_HH (included)")
    ap.add_argument("--hours", type=int, default=3, help="interval in hours (default 3)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--prefix", default="ERA5", help="file prefix, = fg_name in namelist.wps (default ERA5)")
    ap.add_argument("--window", type=float, nargs=4, default=[-40.0, 85.0, 110.0, 359.75],
                    metavar=("LAT0", "LAT1", "LON0", "LON1"),
                    help="lat-lon box to write, degrees (default -40 85 110 359.75); "
                         "LON0 > LON1 crosses 0/360, LON1 - LON0 = 360 is the whole globe")
    ap.add_argument("--rda", default=RDA_DEFAULT, help="ds633.0 top directory (default $RDA or the GLADE copy)")
    ap.add_argument("--nproc", type=int, default=1, help="worker processes (default 1)")
    a = ap.parse_args()

    t0 = dt.datetime.strptime(a.start, "%Y-%m-%d_%H"); t1 = dt.datetime.strptime(a.end, "%Y-%m-%d_%H")
    times = []
    while t0 <= t1:
        times.append(t0); t0 += dt.timedelta(hours=a.hours)
    os.makedirs(a.out, exist_ok=True)
    todo = [t for t in times if not os.path.exists(os.path.join(a.out, f"{a.prefix}:{t:%Y-%m-%d_%H}"))]
    print(f"{len(times)} times, {len(todo)} to write -> {a.out}", flush=True)

    if a.nproc > 1:
        from multiprocessing import Pool
        # group by day so each worker reuses its open daily files
        days = {}
        for t in todo:
            days.setdefault(t.date(), []).append(t)
        # split each day into nproc pieces so small jobs still use every worker
        chunks = []
        for ts in days.values():
            k = max(1, len(ts) // a.nproc)
            chunks += [ts[i:i + k] for i in range(0, len(ts), k)]
        with Pool(a.nproc) as p:
            n = sum(p.map(_work, [(c, a.rda, a.window, a.out, a.prefix) for c in chunks]))
    else:
        n = _work((todo, a.rda, a.window, a.out, a.prefix))
    print(f"wrote {n} files", flush=True)


if __name__ == "__main__":
    main()
