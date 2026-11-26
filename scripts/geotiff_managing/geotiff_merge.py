"""
Merges several GeoTIFF files
-> Use this script if your GeoTIFF is split.
"""

import rasterio
from rasterio.merge import merge

OUTPUT_FILE = "satellite_features_LA_2020.tif"

FILES = [
    "data/geotiff/satellite_features_2020-0000000000-0000000000.tif",
    "data/geotiff/satellite_features_2020-0000009472-0000000000.tif"
]

sources = [rasterio.open(file) for file in FILES]

mosaic, transform = merge(sources)

profile = sources[0].profile.copy()
profile.update(
    height=mosaic.shape[1],
    width=mosaic.shape[2],
    transform=transform
)

with rasterio.open(OUTPUT_FILE, "w", **profile) as dst:
    dst.write(mosaic)
    
    # Preserve band names
    for i, description in enumerate(sources[0].descriptions, start=1):
        dst.set_band_description(i, description)

for src in sources:
    src.close()
    
print("File saved: " + OUTPUT_FILE)