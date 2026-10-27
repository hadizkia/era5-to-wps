# era5-to-wps

Write **WPS intermediate files straight from ERA5 NetCDF**, so WRF's `metgrid`
can use ERA5 without GRIB files or `ungrib`.

It reads the ERA5 0.25° analysis from NCAR's Research Data Archive dataset
**ds633.0** (NetCDF; on NCAR's GLADE file system at
`/glade/campaign/collections/rda/data/d633000`, or a local copy of the same
directory tree) and writes the fields `ungrib` would produce from ERA5 GRIB
with the ECMWF pressure-level Vtable, including ungrib's derived fields
(height from geopotential, relative humidity, snow depth). The output goes
directly into `metgrid.exe`.

Two Python files, no compiled code: `era5_to_wps.py` (the converter) and
`wps_intermediate.py` (reader/writer for the WPS intermediate format,
version 5, lat-lon grids).

Author: Hadi Zanganeh Kia (University of Houston).

## Why

- ERA5 GRIB from the RDA or Copernicus needs `ungrib`, a Vtable, and often
  GRIB2 support in the WPS build. The RDA NetCDF copy of ERA5 is easier to
  get to on NCAR systems but `ungrib` cannot read it.
- With this tool WPS can be built without GRIB2 (`NO_GRIB2`), and only the
  lat-lon window around the WRF domain is written, so the files stay small.

## Requirements

Python 3 with `numpy`, `xarray` and a NetCDF backend (`netCDF4` or `h5netcdf`).
On NCAR's Derecho/Casper the `npl` conda environment has everything:
`module load conda; conda activate npl`.

## Use

```bash
# 3-hourly intermediate files for January 1979, window 40S-85N, 110E-360E (the default)
python3 era5_to_wps.py --start 1979-01-01_00 --end 1979-02-01_00 --hours 3 --out era5_int

# a box over Europe that crosses 0 deg, 16 worker processes, local copy of ds633.0
python3 era5_to_wps.py --start 2010-07-01_00 --end 2010-07-31_18 --hours 6 --out era5_int \
    --window 25 75 -30 45 --nproc 16 --rda /data/rda/d633000
```

| option | meaning |
|---|---|
| `--start`, `--end` | first and last time, `YYYY-MM-DD_HH` (end included) |
| `--hours` | interval (default 3) |
| `--out` | output directory; existing files are skipped, so a rerun continues |
| `--prefix` | file prefix, `ERA5` → `ERA5:1979-01-01_00` (default `ERA5`) |
| `--window LAT0 LAT1 LON0 LON1` | box to write, degrees (default `-40 85 110 359.75`). `LON0 > LON1` crosses 0°/360°; negative longitudes are fine; `0 360` is the whole globe. Make it a few degrees larger than the WRF domain. |
| `--rda` | top directory of ds633.0 (default `$RDA`, else the GLADE path) |
| `--nproc` | worker processes (default 1) |

Then point metgrid at the files in `namelist.wps`
(see `examples/namelist.wps`):

```
&metgrid
 fg_name = 'era5_int/ERA5',
 io_form_metgrid = 2,
/
```

and run `metgrid.exe` as usual, with WPS's standard `METGRID.TBL.ARW`.
Skip `ungrib.exe`. `interval_seconds` in `&share` must match `--hours`.

## Fields written

| level | fields |
|---|---|
| 37 pressure levels (1000–1 hPa) | `HGT` (= z / 9.81), `TT`, `UU`, `VV`, `SPECHUMD`, `RH` (from `SPECHUMD` with ungrib's `compute_rh_spechumd_upa` formula) |
| surface (200100) | `TT` (2 m), `UU`, `VV` (10 m), `RH` (2 m, from the 2 m dew point as ungrib's rrpr), `PSFC`, `SKINTEMP`, `SST`, `SEAICE`, `LANDSEA`, `SOILHGT` (= z_sfc / 9.81), `SNOW` (= sd × 1000), `SNOWH` (= sd × 1000 / ρ_snow), `ST000007` `ST007028` `ST028100` `ST100289`, `SM000007` `SM007028` `SM028100` `SM100289` |
| mean sea level (201300) | `PMSL` |

Missing values (SST and sea ice over land) are written as `-1.e30`, metgrid's
missing value. Winds are earth-relative (regular lat-lon grid).

## Validation

Against the `met_em` file of an existing WRF production run that used `ungrib`
on ERA5 GRIB (same domain, 1979-01-01 00 UTC), after `metgrid`:

- **identical** (difference 0): `UU`, `VV`, `TT`, specific humidity, `PMSL`,
  `SST`, sea ice, snow, soil temperature and moisture, and the heights and
  pressures above the surface;
- **small differences**: relative humidity, 1–3 % in the troposphere;
  land-sea mask, terrain height and surface pressure, at steep terrain only
  (from ERA5's own invariant fields).

The files have since driven WRF month by month (initial and boundary
conditions, SST updates, and 3-hourly grid nudging through `real.exe`'s
`wrffdda`) without problems.

`tests/test_era5_to_wps.py` needs no ERA5: it round-trips the intermediate
format, checks the lat-lon window (including across 0°/360° and the globe),
and runs the converter end to end on a tiny synthetic ds633.0 tree.

```bash
python3 tests/test_era5_to_wps.py
```

## Reading intermediate files in Python

```python
from wps_intermediate import read_file
for r in read_file('era5_int/ERA5:1979-01-01_00'):
    print(r['field'], r['xlvl'], r['data'].shape)
```

## Data credit

ERA5 is produced by ECMWF for the Copernicus Climate Change Service (C3S). If
you use it, cite ERA5 and the RDA dataset:

- Hersbach, H., et al. (2020): The ERA5 global reanalysis. *Q. J. R. Meteorol.
  Soc.*, 146, 1999–2049, <https://doi.org/10.1002/qj.3803>.
- ECMWF (2019): ERA5 Reanalysis (0.25 Degree Latitude-Longitude Grid).
  Research Data Archive at NCAR, <https://doi.org/10.5065/BH6N-5N20>.

The WPS intermediate format is described in the WRF User's Guide (WPS chapter).

## License

[MIT](LICENSE).
