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
import math

PROJECT = "building-height-mapping"  #  Earth Engine cloud project
YEAR = 2020

OUTPUT_DRIVE_FOLDER = "Building Volume Estimation Project"

TARGET_CRS = "EPSG:32611" # UTM zone 11N
PIXEL_SIZE = 20 # meters

BOUNDARY_PATH = Path("boundaries/LA County/County_Boundary.shp")  # file path to target boundary shapefile

ee.Initialize(project=PROJECT)

boundary = gpd.read_file(BOUNDARY_PATH).to_crs("EPSG:4326")  # GeoDataFrame, convert to WGS84

boundary_utm = boundary.to_crs(TARGET_CRS)  # GeoDataFrame, convert to UTM zone 11N
boundary_poly = boundary_utm.geometry.union_all()  # Shapely Polygon

X_MIN, Y_MIN, X_MAX, Y_MAX = boundary_poly.bounds

width = X_MAX - X_MIN
height = Y_MAX - Y_MIN

columns = math.ceil(width / PIXEL_SIZE)
rows = math.ceil(height / PIXEL_SIZE)

#  Fixes pixel size and upper-left origin
GRID_TRANSFORM = [
    PIXEL_SIZE, 0, X_MIN,
    0, -PIXEL_SIZE, Y_MAX
]

# Defines the CRS and affline transform
GRID_PROJECTION = ee.Projection(
    TARGET_CRS,
    GRID_TRANSFORM
)

# Defines the full rectangular raster extent
GRID_REGION = ee.Geometry.Rectangle(
    [
        X_MIN,
        Y_MAX - rows * PIXEL_SIZE,
        X_MIN + columns * PIXEL_SIZE,
        Y_MAX
    ],
    proj=TARGET_CRS,
    geodesic=False
)

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
        .And(scl.neq(7)) # Unclassified
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
        ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1")
        .select("DEM")
    )
    
    native_projection = dem.first().projection()
    
    return (
        dem
        .mosaic() # merges all images into one, to cover the entire area
        .setDefaultProjection(native_projection)
        .rename("COPERNICUS_DEM")
        .clip(region) # limits to study area
    )

def get_dtm():
    dtm = (
        ee.ImageCollection(
            "projects/sat-io/open-datasets/FABDEM"
        )
        .select("b1")
    )
    
    native_projection = dtm.first().projection()

    return (
        dtm
        .mosaic()
        .setDefaultProjection(native_projection)
        .rename("FABDEM")
        .clip(region)
    )
# ------------------------------------------

def aggregate_to_grid(image):
    # Lower resolution for higher-resolution data
    return (
        image
        .reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=16 # Not what the output pixels will be, just a bound that gives room
        )
        .reproject(GRID_PROJECTION) # Set to proper projection
    )

def resample_to_grid(image):
    # Convert lower-resolution data into smaller pixels
    # Does not improve accuracy, only makes new values based on existing data
    return (
        image
        .resample("bilinear")
        .reproject(GRID_PROJECTION) # Set to proper projection
    )

# -----------------------------------------

sentinel2_collection = get_sentinel2()
sentinel2 = create_sentinel2_composite(sentinel2_collection)

# Aggregate 10m bands to 20m grid
s2_10m = aggregate_to_grid(
    sentinel2.select(["B2", "B3", "B4", "B8"])
    .setDefaultProjection(
        sentinel2_collection.first().select("B2").projection() # Use projection of first B2 as default
    )
)

# Algin 20m bands to 20m grid
s2_20m = resample_to_grid(
    sentinel2.select(["B11", "B12"])
    .setDefaultProjection(
        sentinel2_collection.first().select("B11").projection() # Use projection of first B11 as default
    )
)

# Combine values
sentinel2 = s2_10m.addBands(s2_20m)

# Calculate after combining
sentinel2 = add_spectral_indices(sentinel2)

sentinel1_collection = get_sentinel1()
sentinel1 = (
    create_sentinel1_composite(sentinel1_collection)
    .setDefaultProjection(
        sentinel1_collection.first().select("VV").projection()
    )
)

# Convert logarithmic radar backscatter values to linear
sentinel1_linear = ee.Image.constant(10).pow(sentinel1.divide(10)) 

# Convert back to logarithmic dB
sentinel1 = (
    aggregate_to_grid(sentinel1_linear)
    .log10()
    .multiply(10)
    .rename(["VV", "VH"]) # Preserve original names
)

dem = resample_to_grid(get_dem())
dtm = resample_to_grid(get_dtm())


satellite_stack = (
    sentinel2
    .addBands(sentinel1)
    .addBands(dem)
    .addBands(dtm)
).toFloat() # Convert all to Float32, sufficient precision

# --------------------------------------

# Save the GeoTIFF file to your Google Drive
task = ee.batch.Export.image.toDrive(
    image=satellite_stack,
    description=f"satellite_features_{YEAR}",
    folder=OUTPUT_DRIVE_FOLDER,
    fileNamePrefix=f"satellite_features_{YEAR}",
    region=GRID_REGION,
    crs=TARGET_CRS,
    crsTransform=GRID_TRANSFORM,
    maxPixels=1e13, # Allows export of a large raster
    fileFormat="GeoTIFF"
)

task.start()

print("Export started:", task.status())