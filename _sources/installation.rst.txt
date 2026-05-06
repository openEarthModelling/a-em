Installation
============

Requirements
------------

- Python >= 3.9
- OpenCV, NumPy, Pandas, Matplotlib, SciPy
- HyperSpy >= 2.0

Basic Installation
------------------

Install the latest stable release from PyPI:

.. code-block:: bash

   pip install a-em

Optional: SAM Segmentation Support
----------------------------------

For GPU (requires CUDA-capable PyTorch):

.. code-block:: bash

   pip install a-em[sam]

For CPU-only, install CPU-only PyTorch manually after installing ``a-em``:

.. code-block:: bash

   pip install a-em[sam]
   pip install torch --index-url https://download.pytorch.org/whl/cpu

Development Installation
------------------------

Clone the repository and install in editable mode with development dependencies:

.. code-block:: bash

   git clone https://github.com/yourusername/a-em.git
   cd a-em
   pip install -e ".[dev]"
