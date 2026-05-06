Configuration
=============

PipelineConfig
--------------

The :class:`a_em.PipelineConfig` dataclass controls all aspects of the analysis pipeline.

.. code-block:: python

   from a_em import PipelineConfig

   config = PipelineConfig(
       segmentation_backend="traditional",
       analysis_engine="soot",
       output_dir="data/processed",
       clahe_clip=3.0,
       clahe_tile=(16, 16),
       background_kernel=25,
       filter_type="bilateral",
   )

Configuration Parameters
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Parameter
     - Default
     - Description
   * - ``segmentation_backend``
     - ``"traditional"``
     - Segmentation method: ``"traditional"`` or ``"sam_automask"``
   * - ``analysis_engine``
     - ``"soot"``
     - Analysis engine to use
   * - ``output_dir``
     - ``"data/processed"``
     - Directory for output files
   * - ``clahe_clip``
     - ``3.0``
     - CLAHE contrast limit
   * - ``clahe_tile``
     - ``(16, 16)``
     - CLAHE grid tile size
   * - ``background_kernel``
     - ``25``
     - Morphological kernel size for background removal
   * - ``filter_type``
     - ``"bilateral"``
     - Denoising filter: ``"bilateral"`` or ``"gaussian"``

Segmentation Backends
---------------------

**Traditional CV** (``"traditional"``)
   Uses GrabCut and adaptive thresholding. Fast and does not require additional model downloads.

**SAM Auto-Mask** (``"sam_automask"``)
   Uses Meta's Segment Anything Model. Requires ``pip install a-em[sam]`` and a SAM checkpoint file. More robust for complex particle morphologies but slower.

Analysis Engines
----------------

**Soot** (``"soot"``)
   Specialized engine for soot aggregate characterization, computing:

   - Fractal dimension (``Df``)
   - Primary particle diameter
   - Radius of gyration
   - Aggregate morphological metrics
