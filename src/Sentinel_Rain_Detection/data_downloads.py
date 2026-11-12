import ee
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import earthaccess
import xarray as xr

def build_export_image(img_raw):
    """Cast bands to float and recombine into a single export-ready image."""
    vv = img_raw.select('VV').toFloat()
    vh = img_raw.select('VH').toFloat()
    ang = img_raw.select('angle').toFloat()
    return vv.addBands([vh, ang])

def _crop_granule(path, bbox, variables):
    west, south, east, north = bbox
    with xr.open_dataset(path, group="Grid", engine="h5netcdf") as ds:
        ds = ds[variables].sel(lon=slice(west, east), lat=slice(south, north))
        ds = ds.transpose("time", "lat", "lon")
        return ds.load()


def download_imerg_subset(start, end, bbox, variables=("precipitation",),
                          outdir="data/imerg_out", keep_raw=False, merge=False, run="late"):
    """
    start, end : 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' (UTC). A date-only `end` includes that whole day.
    bbox       : (west, south, east, north) in degrees
    run        : "final", "late" or "early"
    Returns    : list of output NetCDF paths (or the merged file path if merge=True)
    """
    RUNS = {"final": "GPM_3IMERGHH", "late": "GPM_3IMERGHHL", "early": "GPM_3IMERGHHE"}

    variables = list(variables)
    short_name = RUNS[run]
    VERSION = "07"

    def _parse(s):
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                pass
        raise ValueError(f"Cannot parse {s}")

    t_start, t_end = _parse(start), _parse(end)
    if len(end) <= 10:
        t_end = t_end + timedelta(days=1) - timedelta(seconds=1)

    outdir = Path(outdir)
    raw_dir = outdir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    day = datetime(t_start.year, t_start.month, t_start.day)
    out_files = []
    enc = {v: {"zlib": True, "complevel": 4} for v in variables}

    while day <= t_end:
        t0 = max(t_start, day)
        t1 = min(t_end, day + timedelta(days=1) - timedelta(seconds=1))
        out_nc = outdir / f"imerg_HH_{day:%Y%m%d}.nc"

        if out_nc.exists():
            print(f"[skip] {out_nc.name}")
            out_files.append(out_nc)
            day += timedelta(days=1)
            continue

        results = earthaccess.search_data(
            short_name=short_name, version=VERSION,
            temporal=(t0.strftime("%Y-%m-%d %H:%M:%S"), t1.strftime("%Y-%m-%d %H:%M:%S")),
        )
        if not results:
            print(f"[{day:%Y-%m-%d}] no granules found")
            day += timedelta(days=1)
            continue

        print(f"[{day:%Y-%m-%d}] downloading {len(results)} granules")
        files = sorted(str(f) for f in earthaccess.download(results, str(raw_dir)))

        pieces = [_crop_granule(f, bbox, variables) for f in files]
        ds_day = xr.concat(pieces, dim="time").sortby("time")
        try:
            ds_day["time"] = ds_day.indexes["time"].to_datetimeindex()
        except Exception:
            pass  # already datetime64
        ds_day.attrs["source"] = f"NASA GES DISC {short_name} V{VERSION}"
        ds_day.attrs["bbox_west_south_east_north"] = str(bbox)
        ds_day.to_netcdf(out_nc, encoding=enc)
        out_files.append(out_nc)

        if not keep_raw:
            for f in files:
                Path(f).unlink(missing_ok=True)
        day += timedelta(days=1)

    if not keep_raw:
        shutil.rmtree(raw_dir, ignore_errors=True)

    if merge and out_files:
        merged = outdir / f"imerg_HH_{t_start:%Y%m%d}_{t_end:%Y%m%d}.nc"
        with xr.open_mfdataset(out_files, combine="by_coords") as ds:
            ds.to_netcdf(merged, encoding=enc)
        return merged
    return out_files