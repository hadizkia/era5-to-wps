"""Read / write WPS intermediate files (format version 5, big-endian Fortran records),
lat-lon projection only.  Used to write ERA5 for metgrid without ungrib."""
import struct
import numpy as np


def _rec(fh, payload):
    fh.write(struct.pack(">i", len(payload))); fh.write(payload); fh.write(struct.pack(">i", len(payload)))


def _s(txt, n):
    return txt.ljust(n)[:n].encode("ascii")


def write_field(fh, hdate, field, units, desc, xlvl, data, startlat, startlon, dlat, dlon,
                map_source="ERA5 reanalysis grid", earth_radius=6367.470215, wind_earth_rel=False):
    """data: (ny, nx) with row 0 at startlat (south-west corner), rows northward."""
    ny, nx = data.shape
    _rec(fh, struct.pack(">i", 5))
    _rec(fh, _s(hdate, 24) + struct.pack(">f", 0.0) + _s(map_source, 32) + _s(field, 9) +
         _s(units, 25) + _s(desc, 46) + struct.pack(">fiii", xlvl, nx, ny, 0))
    _rec(fh, _s("SWCORNER", 8) + struct.pack(">fffff", startlat, startlon, dlat, dlon, earth_radius))
    _rec(fh, struct.pack(">i", 1 if wind_earth_rel else 0))
    _rec(fh, np.ascontiguousarray(data, dtype=">f4").tobytes())   # Fortran (nx,ny) == C (ny,nx)


def read_file(path):
    """-> list of dicts (field, xlvl, units, startloc, startlat, startlon, dlat, dlon, data (ny,nx))."""
    out = []
    with open(path, "rb") as fh:
        def rec():
            h = fh.read(4)
            if not h:
                return None
            n = struct.unpack(">i", h)[0]; b = fh.read(n); fh.read(4); return b
        while True:
            r = rec()
            if r is None:
                break
            b = rec()
            hdate = b[0:24].decode(); field = b[60:69].decode().strip(); units = b[69:94].decode().strip()
            xlvl, nx, ny, iproj = struct.unpack(">fiii", b[140:156])
            p = rec()
            startloc = p[0:8].decode()
            startlat, startlon, dlat, dlon = struct.unpack(">ffff", p[8:24])
            rec()
            d = np.frombuffer(rec(), dtype=">f4").reshape(ny, nx).astype(np.float32)
            out.append(dict(hdate=hdate, field=field, units=units, xlvl=xlvl, iproj=iproj,
                            startloc=startloc, startlat=startlat, startlon=startlon,
                            dlat=dlat, dlon=dlon, data=d))
    return out
