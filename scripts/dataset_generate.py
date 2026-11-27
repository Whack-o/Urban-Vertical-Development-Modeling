
from pathlib import Path
import math
import geopandas as gpd
from shapely.geometry import box
from types import SimpleNamespace
import csv
from tqdm import tqdm
import numpy as np
import shapely
import json

from satellite.satellite_features import SatelliteSampler

# -------------------------------------------------
footprint_path = Path("./data/buildings/LA_Buildings_2020.gdb") # file path to target footprint database
footprint_layer = "LARIAC6_BUILDINGS_2020" # layer name of target footprint database
height_field = "HEIGHT" # field name of building height in target footprint database

boundary_path = Path("./data/boundaries/LA County/County_Boundary.shp") # file path to target boundary shapefile

output_path = Path("./output/training.csv") # csv file to save output
checkpoint_path = Path("./output/checkpoint.json") # file to save processing progress

PIXEL_SIZE = 20 # Meters

PROCESSED_PIXELS = 0 # Number of pixels already processed in previous runs. Use to resume processing

CLEAR_OUTPUT = True # If True, will clear output spreadsheet before starting.
# -------------------------------------------------

print("Preparing geospatial data, please wait...")

BUILDINGS = SimpleNamespace(
    file=footprint_path,
    layer=footprint_layer,
    height_field=height_field
)

BOUNDARY = SimpleNamespace(
    file=boundary_path
)

BOUNDARY.gdf = gpd.read_file(BOUNDARY.file).to_crs("EPSG:32611") # GeoDataFrame, convert to UTM Zone 11N
BOUNDARY.poly = BOUNDARY.gdf.geometry.union_all() # Shapely Polygon

BOUNDARY.X_MIN, BOUNDARY.Y_MIN, BOUNDARY.X_MAX, BOUNDARY.Y_MAX = BOUNDARY.poly.bounds

BOUNDARY.X_MIN = math.floor(BOUNDARY.X_MIN / PIXEL_SIZE) * PIXEL_SIZE # Snap to grid
BOUNDARY.Y_MAX = math.ceil(BOUNDARY.Y_MAX / PIXEL_SIZE) * PIXEL_SIZE

BOUNDARY.width = BOUNDARY.X_MAX - BOUNDARY.X_MIN
BOUNDARY.height = BOUNDARY.Y_MAX - BOUNDARY.Y_MIN
BOUNDARY.columns = math.ceil(BOUNDARY.width / PIXEL_SIZE)
BOUNDARY.rows = math.ceil(BOUNDARY.height / PIXEL_SIZE)

SATELLITE = SatelliteSampler(
    grid_x_min=BOUNDARY.X_MIN,
    grid_y_max=BOUNDARY.Y_MAX,
    pixel_size=PIXEL_SIZE
)

BUILDINGS.gdf = gpd.read_file(
    BUILDINGS.file,
    layer=BUILDINGS.layer
).to_crs("EPSG:32611") # GeoDataFrame of buildings

BUILDINGS.gdf[BUILDINGS.height_field] *= 1200 / 3937

BUILDING_GEOMETRIES = BUILDINGS.gdf.geometry.array

BUILDING_HEIGHTS = BUILDINGS.gdf[
    BUILDINGS.height_field
].to_numpy(dtype=np.float64)


BUILDINGS.spatial_index = BUILDINGS.gdf.sindex # Spatial index

# -------------------------------------------------

total_pixels = BOUNDARY.columns * BOUNDARY.rows

output_path.parent.mkdir(parents=True, exist_ok=True)

# Load checkpoint if resuming
if not CLEAR_OUTPUT and checkpoint_path.exists():
    with open(checkpoint_path, "r") as checkpoint_file:
        PROCESSED_PIXELS = json.load(checkpoint_file)["processed_pixels"]

mode = "w" if CLEAR_OUTPUT or not output_path.exists() else "a"

if PROCESSED_PIXELS >= total_pixels:
    SATELLITE.close()
    raise ValueError("Processing is already complete.")

START_COLUMN = PROCESSED_PIXELS // BOUNDARY.rows
START_ROW = PROCESSED_PIXELS % BOUNDARY.rows

skipped_pixels = 0
checkpoint_interval = 10000

try:
    with open("error_log.txt", "a") as error_log:
        with open(output_path, mode=mode, newline="") as csvfile:
            fieldnames = [
                "x_min", "y_min", "x_max", "y_max",
                "num_buildings", "mean_height",
                "area_weighted_mean_height", "volume_density",
                "satellite_B2", "satellite_B3", "satellite_B4",
                "satellite_B8", "satellite_B11", "satellite_B12",
                "satellite_NDVI", "satellite_NDBI",
                "satellite_VV", "satellite_VH",
                "satellite_COPERNICUS_DEM", "satellite_FABDEM"
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if mode == "w":
                writer.writeheader()

            initial_progress = START_COLUMN * BOUNDARY.rows + START_ROW

            with tqdm(total=total_pixels, initial=initial_progress, desc="Processing Pixels", unit="pixel") as pbar:
                for column in range(START_COLUMN, BOUNDARY.columns):
                    row_start = START_ROW if column == START_COLUMN else 0

                    for row in range(row_start, BOUNDARY.rows):
                        try:
                            pixel = box(
                                BOUNDARY.X_MIN + column * PIXEL_SIZE,
                                BOUNDARY.Y_MAX - (row + 1) * PIXEL_SIZE,
                                BOUNDARY.X_MIN + (column + 1) * PIXEL_SIZE,
                                BOUNDARY.Y_MAX - row * PIXEL_SIZE
                            )

                            #print(f"Processing Pixel at Column: {column}, Row: {row}")

                            if BOUNDARY.poly.covers(pixel):
                                #print("Pixel is fully within boundary")

                                x_min, y_min, x_max, y_max = pixel.bounds

                                indices = BUILDINGS.spatial_index.query(pixel)

                                if len(indices):
                                    areas = shapely.area(
                                        shapely.intersection(BUILDING_GEOMETRIES[indices], pixel)
                                    )

                                    heights = BUILDING_HEIGHTS[indices]

                                    positive = areas > 0
                                    valid = positive & np.isfinite(heights)

                                    valid_heights = heights[valid]
                                    valid_areas = areas[valid]

                                    net_volume = np.dot(valid_heights, valid_areas)
                                    net_area = valid_areas.sum()

                                    num_buildings = np.count_nonzero(positive)

                                    mean_height = valid_heights.mean() if len(valid_heights) else 0
                                    weighted_mean_height = net_volume / net_area if net_area > 0 else 0
                                    volume_density = net_volume / pixel.area

                                else:
                                    num_buildings = 0
                                    mean_height = 0
                                    weighted_mean_height = 0
                                    volume_density = 0

                                satellite_data = SATELLITE.get(row, column)

                                save_data = {
                                    "x_min": x_min,
                                    "y_min": y_min,
                                    "x_max": x_max,
                                    "y_max": y_max,
                                    "num_buildings": num_buildings,
                                    "mean_height": mean_height,
                                    "area_weighted_mean_height": weighted_mean_height,
                                    "volume_density": volume_density
                                }

                                save_data.update(satellite_data)

                                writer.writerow(save_data)

                        except Exception as e:
                            error_log.write(
                                f"Error processing pixel at Column: {column}, Row: {row}. Error: {e}\n"
                            )
                            error_log.flush()
                            skipped_pixels += 1

                        # Update progress after every attempted pixel
                        PROCESSED_PIXELS += 1
                        pbar.update(1)

                        # Save progress periodically
                        if PROCESSED_PIXELS % checkpoint_interval == 0:
                            csvfile.flush()
                            with open(checkpoint_path, "w") as checkpoint_file:
                                json.dump(
                                    {"processed_pixels": PROCESSED_PIXELS},
                                    checkpoint_file
                                )

            # Save final progress
            csvfile.flush()
            with open(checkpoint_path, "w") as checkpoint_file:
                json.dump(
                    {"processed_pixels": PROCESSED_PIXELS},
                    checkpoint_file
                )

    if skipped_pixels > 0:
        print(f"Skipped pixels due to errors: {skipped_pixels}. Check error_log.txt for details.")

finally:
    SATELLITE.close()
