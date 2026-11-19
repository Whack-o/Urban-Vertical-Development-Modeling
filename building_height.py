from pathlib import Path
import pyogrio
import geopandas as gpd

gdb = Path("buildings/LA_Buildings_2020.gdb") # file path to target geodatabase

PIXEL_SIZE = 20 # Meters

def get_intersecting_buildings(x_min, x_max, y_min, y_max):
    # return array of buildings in pixel
    return

def calculate_mean_height(x_min, x_max, y_min, y_max):
    # net height / number of buildings
    return

def calculate_area_weighted_mean_height(x_min, x_max, y_min, y_max):
    # net build volume / net build area
    return

def calculate_volume_density(x_min, x_max, y_min, y_max):
    # net build volume / pixel area
    return

