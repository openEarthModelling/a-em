A-EM Documentation
==================

**A-EM** (Aerosol Electron Microscopy Analyzer) is a Python platform for processing and analyzing aerosol TEM/SEM images. It provides modular, strategy-based pipelines for particle segmentation and quantitative characterization.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   configuration
   api

Features
--------

- **Flexible Architecture** — Registry-based plugin system for segmentation backends and analysis engines.
- **Multiple Segmentation Backends** — Traditional CV and SAM Auto-Mask (Segment Anything Model).
- **Soot Analysis Engine** — Fractal dimension (``Df``) and morphological characterization.
- **Automated Preprocessing** — CLAHE, black-hat background removal, denoising.
- **Batch Processing** — Analyze multiple images in one run.
- **Comprehensive Reporting** — CSV export, labeled masks, and visualization plots.
- **HyperSpy I/O** — Read DM4 and other EM formats with automatic scale calibration.

Quick Example
-------------

.. code-block:: python

   from a_em import PipelineConfig, PipelineExecutor, HyperSpyReader

   signal = HyperSpyReader.read("image.dm4")
   config = PipelineConfig(
       segmentation_backend="traditional",
       analysis_engine="soot",
       output_dir="results/",
   )
   pipeline = PipelineExecutor(config)
   aerosols = pipeline.run(signal)
   print(f"Detected {len(aerosols)} particles")

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
