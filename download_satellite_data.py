"""
Download satellite inputs for the building height model as local GeoTIFFs.

Steps:
  1. Read the boundary file (shapefile, GeoJSON, GeoPackage, anything geopandas reads)
  2. Get its bounds
  3. Download each source for those bounds from Earth Engine, in tiles,
     and save one GeoTIFF per source:
       sentinel2.tif : B2 B3 B4 B8 B11 B12 NDVI NDBI (annual cloud-masked median)
       sentinel1.tif : VV VH (annual median, dB, one orbit direction)
       dem.tif       : DEM (Copernicus GLO-30, meters)

All files share the same 10 m grid in the local UTM zone, so pixel (row, col)
is the same spot in every file.

Setup (one time):
  pip install earthengine-api geopandas rasterio numpy
  earthengine authenticate

Usage:
  python download_satellite_data.py boundary.geojson --year 2023 --project your-gee-project-id
"""

import argparse
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window

SCALE = 10          # meters per pixel
TILE = 512          # pixels per tile side (keeps each request well under EE limits)
NODATA = -9999.0
THREADS = 8
RETRIES = 4


# ---------------------------------------------------------------- EE images

def sentinel2(region, start, end):
    def mask_clouds(img):
        scl = img.select("SCL")
        good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
        return img.updateMask(good).divide(10000)

    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(region)
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
          .map(mask_clouds)
          .select(["B2", "B3", "B4", "B8", "B11", "B12"])
          .median())
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")
    return s2.addBands([ndvi, ndbi])


def sentinel1(region, start, end):
    base = (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"]))
    asc = base.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
    desc = base.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
    # one orbit direction only so viewing geometry is consistent
    return ee.ImageCollection(ee.Algorithms.If(asc.size().gt(0), asc, desc)).median()


def dem(region):
    return (ee.ImageCollection("COPERNICUS/DEM/GLO30")
            .filterBounds(region)
            .select("DEM")
            .mosaic()
            .rename("DEM"))


# ---------------------------------------------------------------- download

def fetch_tile(image, bands, crs, x0, y0, w, h):
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": bands,
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {
                "scaleX": SCALE, "shearX": 0, "translateX": x0,
                "shearY": 0, "scaleY": -SCALE, "translateY": y0,
            },
            "crsCode": crs,
        },
    }
    for attempt in range(RETRIES):
        try:
            arr = ee.data.computePixels(request)
            return np.stack([arr[b] for b in bands]).astype("float32")
        except Exception as e:
            if attempt == RETRIES - 1:
                raise
            wait = 2 ** attempt
            print(f"  tile at ({x0:.0f}, {y0:.0f}) failed ({e}); retrying in {wait}s")
            time.sleep(wait)


def download(image, bounds, crs, out_path):
    minx, miny, maxx, maxy = bounds
    # snap to the 10 m grid so every file lines up exactly
    minx = math.floor(minx / SCALE) * SCALE
    maxy = math.ceil(maxy / SCALE) * SCALE
    width = math.ceil((maxx - minx) / SCALE)
    height = math.ceil((maxy - miny) / SCALE)

    image = image.toFloat().unmask(NODATA)
    bands = image.bandNames().getInfo()

    profile = {
        "driver": "GTiff", "width": width, "height": height,
        "count": len(bands), "dtype": "float32", "crs": crs,
        "transform": from_origin(minx, maxy, SCALE, SCALE),
        "nodata": NODATA, "tiled": True, "blockxsize": 256, "blockysize": 256,
        "compress": "deflate", "BIGTIFF": "IF_SAFER",
    }

    tiles = [(c, r, min(TILE, width - c), min(TILE, height - r))
             for r in range(0, height, TILE) for c in range(0, width, TILE)]
    print(f"{os.path.basename(out_path)}: {width}x{height} px, {len(bands)} bands, {len(tiles)} tiles")

    def job(t):
        c, r, w, h = t
        return t, fetch_tile(image, bands, crs, minx + c * SCALE, maxy - r * SCALE, w, h)

    with rasterio.open(out_path, "w", **profile) as dst:
        for i, name in enumerate(bands, start=1):
            dst.set_band_description(i, name)
        with ThreadPoolExecutor(THREADS) as pool:
            for n, ((c, r, w, h), data) in enumerate(pool.map(job, tiles), start=1):
                dst.write(data, window=Window(c, r, w, h))
                if n % 20 == 0 or n == len(tiles):
                    print(f"  {n}/{len(tiles)} tiles")


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("boundary", help="boundary file (shp, geojson, gpkg, ...)")
    p.add_argument("--year", type=int, required=True, help="match the year of the height labels")
    p.add_argument("--project", required=True, help="Earth Engine cloud project id")
    p.add_argument("--out", default="satellite_data", help="output folder")
    p.add_argument("--buffer", type=float, default=100, help="meters of padding around the bounds")
    args = p.parse_args()

    ee.Initialize(project=args.project)
    os.makedirs(args.out, exist_ok=True)

    boundary = gpd.read_file(args.boundary)
    utm = boundary.estimate_utm_crs()
    crs = f"EPSG:{utm.to_epsg()}"
    minx, miny, maxx, maxy = boundary.to_crs(utm).total_bounds
    b = args.buffer
    bounds = (minx - b, miny - b, maxx + b, maxy + b)

    w, s, e, n = boundary.to_crs(4326).total_bounds
    region = ee.Geometry.Rectangle([w, s, e, n])
    start, end = f"{args.year}-01-01", f"{args.year + 1}-01-01"

    print(f"Bounds in {crs}: {[round(v) for v in bounds]}")
    download(sentinel2(region, start, end), bounds, crs, os.path.join(args.out, "sentinel2.tif"))
    download(sentinel1(region, start, end), bounds, crs, os.path.join(args.out, "sentinel1.tif"))
    download(dem(region), bounds, crs, os.path.join(args.out, "dem.tif"))
    print(f"Done. Files in {args.out}/")


if __name__ == "__main__":
    main()
