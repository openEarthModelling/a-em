from setuptools import setup, find_packages

setup(
    name="atem_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python",
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
    ],
    author="ATEM Team",
    description="A Python package for analyzing Aerosol Transmission Electron Microscopy (TEM) images.",
    python_requires=">=3.7",
)
