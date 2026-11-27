import rasterio
import geopandas as gpd
import math

PIXEL_SIZE = 20
TARGET_CRS = "EPSG:32611"

BOUNDARY_PATH = "data/boundaries/LA County/County_Boundary.shp"
RASTER_PATH = "data/geotiff/satellite_features_LA_2020.tif"

boundary = gpd.read_file(BOUNDARY_PATH).to_crs(TARGET_CRS)
boundary_poly = boundary.geometry.union_all()

X_MIN, Y_MIN, X_MAX, Y_MAX = boundary_poly.bounds

X_MIN = math.floor(X_MIN / PIXEL_SIZE) * PIXEL_SIZE
Y_MAX = math.ceil(Y_MAX / PIXEL_SIZE) * PIXEL_SIZE

with rasterio.open(RASTER_PATH) as ds:
    t = ds.transform

    print("\n--- RASTER ---")
    print("CRS:", ds.crs)
    print("Transform:", t)
    print("Dimensions:", ds.width, "x", ds.height)
    print("Bands:", ds.count)
    print("Pixel size:", t.a, t.e)

    print("\n--- EXPECTED GRID ---")
    print("CRS:", TARGET_CRS)
    print("X_MIN:", X_MIN)
    print("Y_MAX:", Y_MAX)
    print("Pixel size:", PIXEL_SIZE)

    col_offset = (X_MIN - t.c) / PIXEL_SIZE
    row_offset = (t.f - Y_MAX) / PIXEL_SIZE

    print("\n--- ALIGNMENT ---")
    print("Column offset:", col_offset)
    print("Row offset:", row_offset)
    print("Column remainder:", col_offset - round(col_offset))
    print("Row remainder:", row_offset - round(row_offset))

    print("\n--- ALIGNED? ---")
    print("X:", math.isclose(col_offset, round(col_offset), abs_tol=1e-6))
    print("Y:", math.isclose(row_offset, round(row_offset), abs_tol=1e-6))