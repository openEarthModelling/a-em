# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2024-05-06

### Added

- SAM Auto-Mask segmentation backend (`SAMAutoMaskSegmenter`) for improved particle detection using Segment Anything Model (SAM).
- Post-filtering pipeline for SAM masks (area filtering, nested mask resolution).
- `extras_require` support for optional SAM dependencies (`pip install a-em[sam]`).
- Registry-based plugin architecture for segmentation backends and analysis engines.
- Comprehensive test suite covering segmentation, analysis, and pipeline workflows.

### Changed

- Refactored project structure into a flat package layout (`a-em/`).
- Migrated configuration to `PipelineConfig` dataclass.
- Adopted modern packaging standards with `pyproject.toml`.

## [0.1.0] - 2024-03-20

### Added

- Initial release of A-EM.
- HyperSpy-based DM4 image I/O with automatic scale calibration.
- Traditional CV segmentation (GrabCut, adaptive thresholding) for aerosol particles.
- Soot analysis engine with fractal dimension (`Df`) calculation.
- Batch processing support for multiple images.
- CSV export and labeled mask visualization.
