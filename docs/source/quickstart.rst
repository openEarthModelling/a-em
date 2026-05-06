Quick Start
===========

Single Image Analysis
---------------------

The simplest way to analyze a single image is through the :class:`a_em.PipelineExecutor`:

.. code-block:: python

   from a_em import PipelineConfig, PipelineExecutor, HyperSpyReader

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

Batch Processing
----------------

Process multiple images in one run using :class:`a_em.BatchProcessor`:

.. code-block:: python

   from a_em import PipelineConfig, BatchProcessor

   config = PipelineConfig(
       segmentation_backend="traditional",
       analysis_engine="soot",
       output_dir="batch_results/",
   )

   processor = BatchProcessor(config)
   results = processor.process_directory("data/raw/")

Accessing Results
-----------------

Each detected particle is represented as an :class:`a_em.AerosolObject`:

.. code-block:: python

   for obj in aerosols:
       print(f"ID: {obj.aerosol_id}")
       print(f"Bounding box: {obj.bbox}")
       print(f"Metrics: {obj.metrics}")
