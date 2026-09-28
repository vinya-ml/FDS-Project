# Foundations Of Data Science - Project

This folder contains the complete project pipeline, scripts, analysis results, visualizations, and datasets.

## Datasets
Due to GitHub file size restrictions (100 MB per file limit), the CSV datasets in this folder have been compressed using gzip (`.csv.gz`).

### Loading in Python (Pandas)
Pandas supports reading `.csv.gz` files natively without manual decompression:
```python
import pandas as pd

# Directly load any compressed dataset
df = pd.read_csv("analysis_mandi.csv.gz")
```

### Decompressing to raw `.csv`
If you need the raw uncompressed `.csv` files:
- **Terminal (gzip)**: `gzip -d filename.csv.gz`
- **Windows Tools**: 7-Zip, WinRAR, or standard archive extractors can extract them directly.
