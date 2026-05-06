# A-EM (Aerosol Electron Microscopy Analyzer)

**A-EM** (Aerosol Transmission Electron Microscopy Analyzer) is a Python platform for processing and analyzing aerosol TEM/SEM images. It provides modular, strategy-based pipelines for particle segmentation and quantitative characterization.

## Features

- **Flexible Architecture** — Registry-based plugin system for segmentation backends and analysis engines.
- **Multiple Segmentation Backends**:
  - Traditional CV (GrabCut, adaptive thresholding)
  - SAM Auto-Mask (Segment Anything Model for improved detection)
- **Soot Analysis Engine** — Specialized fractal dimension (`Df`) and morphological characterization.
- **Automated Preprocessing** — CLAHE, black-hat background removal, denoising.
- **Batch Processing** — Analyze multiple images in one run.
- **Comprehensive Reporting** — CSV data export, labeled masks, and visualization plots.
- **HyperSpy I/O** — Read DM4 and other EM formats with automatic scale calibration.

## Installation

```bash
pip install a-em
```

### Optional: SAM segmentation support

For GPU (requires CUDA-capable PyTorch):
```bash
pip install a-em[sam]
```

For CPU-only:
```bash
pip install a-em[sam]
# Then install CPU-only PyTorch manually:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

```python
from a_em import (
    PipelineConfig, PipelineExecutor, HyperSpyReader
)

# Load a DM4 image
signal = HyperSpyReader.read("path/to/image.dm4")

# Configure the pipeline
config = PipelineConfig(
    segmentation_backend="traditional",  # or "sam_automask"
    analysis_engine="soot",
    output_dir="results/",
)

# Run the pipeline
pipeline = PipelineExecutor(config)
aerosols = pipeline.run(signal)

print(f"Detected {len(aerosols)} particles")
```

## Project Structure

```
a_em/
├── a_em/          # Core package
│   ├── __init__.py
│   ├── config.py           # Pipeline configuration
│   ├── core.py             # AerosolObject model
│   ├── io.py               # HyperSpy-based image I/O
│   ├── preprocess.py       # Image enhancement
│   ├── pipeline.py         # Single-image pipeline executor
│   ├── batch.py            # Batch processor
│   ├── reporter.py         # Results export & visualization
│   ├── segmentation/       # Segmentation backends
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── traditional.py
│   │   ├── sam_automask.py
│   │   └── legacy.py
│   └── analysis/           # Analysis engines
│       ├── base.py
│       ├── registry.py
│       └── soot.py
├── tests/                  # Test suite
├── examples/               # Example scripts
├── data/                   # Data directory (not included)
├── pyproject.toml
└── README.md
```

## Configuration

`PipelineConfig` supports the following options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segmentation_backend` | `"traditional"` | `"traditional"` or `"sam_automask"` |
| `analysis_engine` | `"soot"` | `"soot"` (extensible) |
| `output_dir` | `"data/processed"` | Directory for results |
| `clahe_clip` | `3.0` | CLAHE contrast limit |
| `clahe_tile` | `(16, 16)` | CLAHE grid size |
| `background_kernel` | `25` | Morphological kernel size |
| `filter_type` | `"bilateral"` | `"bilateral"` or `"gaussian"` |

## Testing

```bash
pytest
```

## License

MIT License — see [LICENSE](LICENSE).

## Citation

If you use A-EM in your research, please cite this repository.
