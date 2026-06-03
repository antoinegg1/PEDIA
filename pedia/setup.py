from setuptools import setup, find_packages

setup(
    name="pedia",
    version="0.1.0",
    package_dir={"": ".."},
    packages=[p for p in find_packages("..") if p == "pedia" or p.startswith("pedia.")],
)
