# Sample Input

This directory should contain sample Sentinel-1 SAR tiles for running the demo and inference.

## How to obtain sample tiles

### Option 1: From Sen1Floods11
Download a few tiles from the Sen1Floods11 dataset:
```bash
gsutil cp gs://sen1floods11/S1Hand/India_103757_S1Hand.tif ./sample_input/
gsutil cp gs://sen1floods11/S1Hand/Bolivia_103757_S1Hand.tif ./sample_input/
```

### Option 2: From Copernicus Browser
1. Visit https://browser.dataspace.copernicus.eu/
2. Search for Sentinel-1 GRD products over a known flood region
3. Download a tile and place it here

### Option 3: Synthetic (no download needed)
The inference script and Gradio app can generate synthetic SAR tiles:
```bash
python infer.py --synthetic --output-dir ./sample_output
python app.py  # Toggle "synthetic" in the UI
```

## Expected tile format
- GeoTIFF (`.tif`)
- Sentinel-1 SAR with at least VV and VH bands (2 channels)
- Any resolution (will be resized to 224×224)
- Projection: EPSG:4326 (WGS 84) preferred
- dB scale (backscatter typically in [-30, 0] range)

## Note
Do **not** commit large files (>10 MB) to this directory. Link to external sources instead.
