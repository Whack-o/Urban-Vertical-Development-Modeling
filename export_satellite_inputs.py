"""
Export satellite input stacks for the building height model.

Bands exported (all on a 10 m grid, EPSG:4326 by default):
  Sentinel-2: B2, B3, B4, B8 (10 m), B11, B12 (20 m, resampled), NDVI, NDBI
  Sentinel-1: VV, VH (ascending orbit, annual median)
  Copernicus DEM GLO-30: DEM

Setup (one time):
  pip install earthengine-api
  earthengine authenticate

Usage:
  python export_satellite_inputs.py
Files land in your Google Drive folder "building_height_inputs".
Edit AREAS and YEAR below. YEAR should match the year of your height labels.
"""

import ee

PROJECT = "your-gee-project-id"  # your Earth Engine cloud project
YEAR = 2023
DRIVE_FOLDER = "building_height_inputs"

# name: [west, south, east, north]  (lon/lat). Swap for wherever labels exist.
AREAS = {
    "houston": [-95.45, 29.70, -95.30, 29.80],
    "katy": [-95.85, 29.74, -95.72, 29.82],
    "dallas": [-96.85, 32.74, -96.75, 32.82],
}

ee.Initialize(project=PROJECT)


def s2_composite(region, start, end):
    def mask_clouds(img):
        scl = img.select("SCL")
        # keep vegetation, bare soil, water, unclassified; drop clouds/shadow
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


def s1_composite(region, start, end):
    base = (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"]))
    asc = base.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
    desc = base.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
    # Use ascending if available, otherwise descending (one direction only,
    # so the viewing geometry stays consistent within an area).
    chosen = ee.ImageCollection(ee.Algorithms.If(asc.size().gt(0), asc, desc))
    return chosen.median()


def dem(region):
    return (ee.ImageCollection("COPERNICUS/DEM/GLO30")
            .filterBounds(region)
            .select("DEM")
            .mosaic()
            .rename("DEM"))


def build_stack(region, year):
    start, end = f"{year}-01-01", f"{year}-12-31"
    return (s2_composite(region, start, end)
            .addBands(s1_composite(region, start, end))
            .addBands(dem(region))
            .toFloat()
            .clip(region))


for name, bbox in AREAS.items():
    region = ee.Geometry.Rectangle(bbox)
    task = ee.batch.Export.image.toDrive(
        image=build_stack(region, YEAR),
        description=f"{name}_{YEAR}_inputs",
        folder=DRIVE_FOLDER,
        region=region,
        scale=10,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    print(f"Started export: {name}_{YEAR}_inputs")

print("Check progress at https://code.earthengine.google.com/tasks")
