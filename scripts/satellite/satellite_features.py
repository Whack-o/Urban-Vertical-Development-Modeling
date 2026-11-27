"""
USES A SINGULAR, MULTI-BAND GEOTIFF


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


import numpy as np
import rasterio
from pyproj import CRS

SOURCE_PATH = "data/geotiff/satellite_features_LA_2020.tif"
POLYGON_CRS = "EPSG:32611"


class SatelliteSampler:
    def __init__(self, grid_x_min, grid_y_max, pixel_size):
        self.ds = rasterio.open(SOURCE_PATH)

        if CRS.from_user_input(self.ds.crs) != CRS.from_user_input(POLYGON_CRS):
            raise ValueError("Satellite raster CRS does not match training grid.")

        t = self.ds.transform

        if (
            not np.isclose(t.a, pixel_size)
            or not np.isclose(t.e, -pixel_size)
            or not np.isclose(t.b, 0)
            or not np.isclose(t.d, 0)
        ):
            raise ValueError("Satellite raster resolution or orientation does not match training grid.")

        col_offset = (grid_x_min - t.c) / pixel_size
        row_offset = (t.f - grid_y_max) / pixel_size

        if not np.isclose(col_offset, round(col_offset), atol=1e-6):
            raise ValueError("Training grid is not aligned with satellite raster.")

        if not np.isclose(row_offset, round(row_offset), atol=1e-6):
            raise ValueError("Training grid is not aligned with satellite raster.")

        self.col_offset = round(col_offset)
        self.row_offset = round(row_offset)

        self.names = [f"satellite_{d}" for d in self.ds.descriptions]

        estimated_gb = (
            self.ds.width * self.ds.height * self.ds.count
            * np.dtype(self.ds.dtypes[0]).itemsize / 1024**3
        )

        print(f"Loading satellite raster into memory: {estimated_gb:.2f} GB")
        self.data = self.ds.read()

    def get(self, row, column):
        r = self.row_offset + row
        c = self.col_offset + column

        if not (0 <= r < self.ds.height and 0 <= c < self.ds.width):
            return {name: np.nan for name in self.names}

        values = self.data[:, r, c]
        valid = np.isfinite(values)

        if self.ds.nodata is not None:
            valid &= values != self.ds.nodata

        return {
            name: float(value) if is_valid else np.nan
            for name, value, is_valid in zip(self.names, values, valid)
        }

    def close(self):
        self.ds.close()
