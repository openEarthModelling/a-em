# Minimal setup.py for backward compatibility.
# All packaging configuration lives in pyproject.toml.
from setuptools import setup, find_packages

setup(
    name="a_em",
    packages=find_packages(exclude=["tests*", "examples*"]),
    package_data={"a_em": ["py.typed"]},
)
