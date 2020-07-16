from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages

def _requires_from_file(filename):
    return open(filename).read().splitlines()

with open("README.md", "r") as f:
    long_description = f.read()

print(find_packages("gp"))
setup(
    name="GaussianProcess",
    version="0.5.0",
    license="MIT License",
    description="A Python Package for Gaussian Process Regression.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Yu Kiyokawa",
    author_email='dummn.marionette.7surspecies@gmail.com',
    url="https://github.com/JohnYKiyo/GaussianProcess",
    keywords='gaussian process',
    python_requires=">=3.6.0",
    packages = [s.replace('gp','GaussianProcess') for s in find_packages('.')],
    package_dir={"GaussianProcess": "gp"},
    py_modules=[splitext(basename(path))[0] for path in glob('gp/*.py')],
    install_requires=[
        'jax>=0.1.57',
        'jaxlib>=0.1.37'
    ]
)
