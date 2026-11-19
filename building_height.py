from pathlib import Path
import math
import geopandas as gpd
from shapely.geometry import box
from types import SimpleNamespace
import csv
from tqdm import tqdm

# -------------------------------------------------
footprint_path = Path("buildings/LA_Buildings_2020.gdb") # file path to target footprint database
footprint_layer = "LARIAC6_BUILDINGS_2020" # layer name of target footprint database
height_field = "HEIGHT" # field name of building height in target footprint database

boundary_path = Path("boundaries/LA County/County_Boundary.shp") # file path to target boundary shapefile

output_path = Path("output/training.csv") # csv file to save output

PIXEL_SIZE = 20 # Meters

START_COLUMN = 0
START_ROW = 0

CLEAR_OUTPUT = False # If True, will clear output spreadsheet before starting.
# -------------------------------------------------

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

BOUNDARY.width = BOUNDARY.X_MAX - BOUNDARY.X_MIN
BOUNDARY.height = BOUNDARY.Y_MAX - BOUNDARY.Y_MIN
BOUNDARY.columns = math.ceil(BOUNDARY.width / PIXEL_SIZE)
BOUNDARY.rows = math.ceil(BOUNDARY.height / PIXEL_SIZE)

BUILDINGS.gdf = gpd.read_file(
    BUILDINGS.file,
    layer=BUILDINGS.layer
).to_crs("EPSG:32611") # GeoDataFrame of buildings

BUILDINGS.gdf[BUILDINGS.height_field] *= 1200 / 3937 # Convert the height from feet to meters (1 foot = 0.3048 meters, 1200/3937 is approximately 0.3048)

BUILDINGS.spatial_index = BUILDINGS.gdf.sindex # Spatial index

def get_intersecting_buildings(pixel):
    indices = BUILDINGS.spatial_index.query(pixel, predicate="intersects") 
    intersection_buildings = BUILDINGS.gdf.iloc[indices]
    return intersection_buildings

def calculate_mean_height(buildings, intersection_areas):
    valid = (intersection_areas > 0) & buildings[BUILDINGS.height_field].notna()
    return buildings.loc[valid, BUILDINGS.height_field].mean() if valid.any() else 0

def calculate_area_weighted_mean_height(buildings, intersection_areas):
    valid = (intersection_areas > 0) & buildings[BUILDINGS.height_field].notna()
    net_volume = (buildings.loc[valid, BUILDINGS.height_field] * intersection_areas[valid]).sum()
    net_area = intersection_areas[valid].sum()
    return net_volume / net_area if net_area > 0 else 0

def calculate_volume_density(pixel, buildings, intersection_areas):
    valid = (intersection_areas > 0) & buildings[BUILDINGS.height_field].notna()
    net_volume = (buildings.loc[valid, BUILDINGS.height_field] * intersection_areas[valid]).sum()
    return net_volume / pixel.area if pixel.area > 0 else 0

# -------------------------------------------------

total_pixels = BOUNDARY.columns * BOUNDARY.rows

mode = "W" if CLEAR_OUTPUT or not output_path.exists() else "a"

with open(output_path, mode=mode, newline="") as csvfile:
    fieldnames = ["x_min", "y_min", "x_max", "y_max", "num_buildings", "mean_height", "area_weighted_mean_height", "volume_density"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    if mode == "w":
        writer.writeheader()
    
    initial_progress = START_COLUMN * BOUNDARY.rows + START_ROW

    with tqdm(total=total_pixels, initial=initial_progress, desc="Processing Pixels", unit="pixel") as pbar:
        for column in range(START_COLUMN, BOUNDARY.columns):
            row_start = START_ROW if column == START_COLUMN else 0
            for row in range(row_start, BOUNDARY.rows):
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
                    
                    interesting_buildings = get_intersecting_buildings(pixel)
                    intersection_areas = interesting_buildings.geometry.intersection(pixel).area
                    
                    save_data = {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max,
                        "num_buildings": (intersection_areas > 0).sum(), # only count positive-area intersections
                        "mean_height": calculate_mean_height(interesting_buildings, intersection_areas),
                        "area_weighted_mean_height": calculate_area_weighted_mean_height(interesting_buildings, intersection_areas),
                        "volume_density": calculate_volume_density(pixel, interesting_buildings, intersection_areas)
                    }
                    
                    writer.writerow(save_data)
                else:
                    #print("Pixel is outside of boundary, skipping...")
                    pass
                
                pbar.update(1)