"""
Read satellite values for one polygon (e.g. a label pixel) from the local
GeoTIFFs made by download_satellite_data.py. No network calls, so it is fast
enough to run millions of times.

Use in building_heights.py:

    from satellite_features import get_satellite_data

    row = {...building height stuff...}
    row.update(get_satellite_data(polygon))   # polygon = shapely Polygon
    writer.writerow(row)

Returns the mean of each band inside the polygon, e.g.
    {"s2_B2": 0.051, ..., "s2_NDBI": 0.12, "s1_VV": -8.3, "s1_VH": -15.1, "dem_DEM": 41.2}
Bands with no valid pixels come back as NaN.

Set POLYGON_CRS to whatever CRS building_heights.py uses for its polygons.
"""

import os

import numpy as np
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import geometry_mask, geometry_window
from rasterio.errors import WindowError
import shapely
from shapely.geometry import mapping

DATA_DIR = "satellite_data"
POLYGON_CRS = "EPSG:4326"
SOURCES = {"s2": "sentinel2.tif", "s1": "sentinel1.tif", "dem": "dem.tif"}


class SatelliteSampler:
    def __init__(self, data_dir=DATA_DIR, polygon_crs=POLYGON_CRS):
        self.sources = []
        for prefix, fname in SOURCES.items():
            ds = rasterio.open(os.path.join(data_dir, fname))
            names = [f"{prefix}_{d or i}" for i, d in enumerate(ds.descriptions, start=1)]
            to_raster = None
            if CRS.from_user_input(polygon_crs) != CRS.from_user_input(ds.crs):
                tf = Transformer.from_crs(polygon_crs, ds.crs, always_xy=True)
                to_raster = lambda xy, tf=tf: np.column_stack(tf.transform(xy[:, 0], xy[:, 1]))
            self.sources.append((ds, names, to_raster))

    def get(self, polygon):
        out = {}
        for ds, names, to_raster in self.sources:
            geom = shapely.transform(polygon, to_raster) if to_raster else polygon
            try:
                win = geometry_window(ds, [mapping(geom)])
            except WindowError:  # polygon is outside the downloaded area
                out.update({n: np.nan for n in names})
                continue
            data = ds.read(window=win)
            inside = geometry_mask([mapping(geom)], out_shape=data.shape[1:],
                                   transform=ds.window_transform(win),
                                   invert=True, all_touched=True)
            for band, name in zip(data, names):
                vals = band[inside]
                vals = vals[vals != ds.nodata] if ds.nodata is not None else vals
                out[name] = float(vals.mean()) if vals.size else np.nan
        return out

    def close(self):
        for ds, _, _ in self.sources:
            ds.close()


_sampler = None


def get_satellite_data(polygon):
    """Mean satellite band values inside `polygon` (shapely geometry in POLYGON_CRS)."""
    global _sampler
    if _sampler is None:
        _sampler = SatelliteSampler()
    return _sampler.get(polygon)
