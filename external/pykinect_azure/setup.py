from setuptools import setup, find_packages

# read the contents of your README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='pykinect_azure',
    version='0.0.1',
    license='MIT',
    description='Python library to control the Azure Kinect',
    author='Ibai Gorordo & Vanessa Wirth (forked)',
    url='https://github.com/ibaiGorordo/pyKinectAzure',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
    ],
)
