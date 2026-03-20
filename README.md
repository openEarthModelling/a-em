# ATEM Analyzer

**ATEM Analyzer** (Aerosol Transmission Electron Microscopy Analyzer) is a generalized platform for processing and analyzing aerosol TEM images.

## Features
- **Flexible Architecture**: Strategy-based analysis engines (e.g., Soot, future extensions).
- **Automated Preprocessing**: CLAHE, background removal, and denoising.
- **Robust Segmentation**: Supports GrabCut and automated thresholding.
- **Fractal Analysis**: Specialized engine for soot aggregate fractal dimension (Df) calculation.
- **Comprehensive Reporting**: CSV data export, labeled masks, and visualization plots.

## Project Structure
- `src/`: Core logic and modules.
  - `atem_analysis/`: Analysis strategy engines.
  - `atem_io.py`: Image reading and scale detection.
  - `atem_preprocess.py`: Image enhancement.
  - `atem_segmentation.py`: Mask extraction.
  - `atem_core.py`: Object model for aerosols.
  - `atem_reporter.py`: Results visualization and export.
- `data/`:
  - `raw/`: Input TEM images.
  - `processed/`: Analysis results.

## Usage
Run the analyzer from the project root:
```bash
PYTHONPATH=src python3 src/main.py --input data/raw --output data/processed --engine soot
```

### Options:
- `--input`: Path to a file or directory of images.
- `--output`: Directory to save results.
- `--engine`: Type of analysis to perform (currently only 'soot' is supported).
