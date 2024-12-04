from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
# long_description = (this_directory / "README.md").read_text()

setup(
    name='maroon',
    version='0.0.1',
    license='MIT',
    description='Python library to display dataset information from maroon',
    author='Vanessa Wirth',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
        "scipy",
        "open3d",
        "nibabel",
        "matplotlib",
        "serial",
        "pycuda",
        "astropy",
        "plotly",
        "pymeshlab",
        "pymesh"
    ],
)
