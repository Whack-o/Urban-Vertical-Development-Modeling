"""
Generates a GeoTIFF file using training data
Increase the parameters in METRICS with the column headers in the training data to increase number of bands in output file.
"""


from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

# -------------------------------------------------
input_path = Path("output/training.csv")
output_path = Path("output/training_data.tif")

PIXEL_SIZE = 20
CRS = "EPSG:32611"

METRICS = {
    "Mean Building Height": "mean_height",
    "Area-Weighted Mean Height": "area_weighted_mean_height",
    "Volume Density": "volume_density",
    "Building Count": "num_buildings"
}
# -------------------------------------------------

print("Reading training data...")
df = pd.read_csv(input_path)

print(f"Loaded {len(df):,} pixels.")

# Determine raster dimensions and origin
x_min = df["x_min"].min()
y_max = df["y_max"].max()

columns = round((df["x_max"].max() - x_min) / PIXEL_SIZE)
rows = round((y_max - df["y_min"].min()) / PIXEL_SIZE)

print(f"Raster dimensions: {columns:,} x {rows:,}")

# Calculate pixel positions
col_indices = np.rint(
    (df["x_min"].to_numpy() - x_min) / PIXEL_SIZE
).astype(int)

row_indices = np.rint(
    (y_max - df["y_max"].to_numpy()) / PIXEL_SIZE
).astype(int)

# Create multiband raster
raster = np.full(
    (len(METRICS), rows, columns),
    np.nan,
    dtype=np.float32
)

for band, (name, field) in enumerate(METRICS.items()):
    print(f"Processing: {name}")

    values = df[field].to_numpy(dtype=np.float32)

    raster[band, row_indices, col_indices] = values

# Define geographic transform
transform = from_origin(
    x_min,
    y_max,
    PIXEL_SIZE,
    PIXEL_SIZE
)

# Export GeoTIFF
print("Writing GeoTIFF...")

with rasterio.open(
    output_path,
    "w",
    driver="GTiff",
    height=rows,
    width=columns,
    count=len(METRICS),
    dtype="float32",
    crs=CRS,
    transform=transform,
    nodata=np.nan,
    compress="lzw"
) as dst:

    for band, (name, field) in enumerate(METRICS.items(), start=1):
        dst.write(raster[band - 1], band)
        dst.set_band_description(band, name)

print(f"Finished: {output_path}")
