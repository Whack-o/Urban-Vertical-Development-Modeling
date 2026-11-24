"""
    Resolution Information:
    Sentinel-2 B2, B3, B4, B8: 10 m
    Sentinel-2 B11, B12: 20 m
    Sentinel-1 VV, VH: ~10 m
    Copernicus DEM GLO-30: ~30 m
"""

from pathlib import Path
import geopandas as gpd
import ee

PROJECT = "building-height-mapping"  #  Earth Engine cloud project
YEAR = 2020

BOUNDARY_PATH = Path("boundaries/LA County/County_Boundary.shp")  # file path to target boundary shapefile

ee.Initialize(project=PROJECT)

boundary = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")  # GeoDataFrame, convert to WGS84

west, south, east, north = boundary.total_bounds  # Get the bounding box

region = ee.Geometry.Rectangle([west, south, east, north])  # Create an Earth Engine rectangle geometry

def get_sentinel2():
    start_date = f"{YEAR}-01-01"
    end_date = f"{YEAR + 1}-01-01" # Exclusive end date
    
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(start_date, end_date)
    )
    
    return collection

def mask_s2_clouds(image):
    scl = image.select("SCL")
    
    clear = (
        scl.neq(0) # No data
        .And(scl.neq(1)) # Saturated or defective
        .And(scl.neq(3)) # Cloud shadows
        .And(scl.neq(8)) # Cloud medium probability
        .And(scl.neq(9)) # Cloud high probability
        .And(scl.neq(10)) # Thin cirrus
        .And(scl.neq(11)) # Snow or ice
    )

    
    return image.updateMask(clear)

def create_sentinel2_composite(collection):
    """
        B2: Blue (490 nm)
        B3: Green (560 nm)
        B4: Red (665 nm)
        B8: Near-infrared (842 nm)
        B11: Short-wave infrared 1 (1610 nm)
        B12: Short-wave infrared 2 (2190 nm)
    """
    
    bands = ["B2", "B3", "B4", "B8", "B11", "B12"]
    
    composite = (
        collection
        .map(mask_s2_clouds) # applies cloud mask
        .select(bands) # keeps only bands of interest
        .median() # calcualtes median value for each band at each pixel
        .multiply(0.0001) # converts to scaled reflectance values
        .clip(region) # limits to study area
    )
    
    return composite
    
def add_spectral_indices(image):
    ndvi = image.expression(
        "(NIR - RED) / (NIR + RED)", {
            "NIR": image.select("B8"),
            "RED": image.select("B4")
        }
    ).rename("NDVI")
    
    ndbi = image.expression(
        "(SWIR - NIR) / (SWIR + NIR)", {
            "SWIR": image.select("B11"),
            "NIR": image.select("B8")
        }
    ).rename("NDBI")
    
    return image.addBands([ndvi, ndbi])

def get_sentinel1():
    start_date = f"{YEAR}-01-01"
    end_date = f"{YEAR + 1}-01-01" # Exclusive end date
    
    collection = (
        ee.ImageCollection("COPERNICUS/S1_GRD") # Sentinel-1 Ground Range Detected
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VV"
            )
        )
        .filter(
            ee.Filter.listContains(
                "transmitterReceiverPolarisation", "VH"
            )
        )
        .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING"))
        .select(["VV", "VH"])
    )
    
    return collection

def create_sentinel1_composite(collection):
    composite = (
        collection
        .median() # calculates median value for each band at each pixel
        .clip(region) # limits to study area
    )
    
    return composite
    
def get_dem():
    # Only uses 2011-2015 data
    # Consider removing DEM later, and only using DTM
    
    dem = (
        ee.ImageCollection("COPERNICUS/DEM/GLO30")
        .select("DEM")
        .rename("COPERNICUS_DEM")
        .mosaic() # merges all images into one, to cover the entire area
        .clip(region) # limits to study area
    )
    
    return dem

def get_dtm():
    dtm = (
        ee.ImageCollection(
            "projects/sat-io/open-datasets/FABDEM"
        )
        .mosaic()
        .select("b1")
        .rename("FABDEM")
        .clip(region)
    )

    return dtm
# ------------------------------------------

sentinel2_collection = get_sentinel2()

sentinel2 = create_sentinel2_composite(sentinel2_collection)
sentinel2 = add_spectral_indices(sentinel2)

sentinel1_collection = get_sentinel1()
sentinel1 = create_sentinel1_composite(sentinel1_collection)

dem = get_dem()
dtm = get_dtm()


satellite_stack = (
    sentinel2
    .addBands(sentinel1)
    .addBands(dem)
    .addBands(dtm)
)