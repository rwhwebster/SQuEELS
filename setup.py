from setuptools import setup, find_packages
import codecs
import os
import sys
import re

here = os.path.abspath(os.path.dirname(__file__))

# parse version from init
# from: https://github.com/pypa/pip/blob/master/setup.py
here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name='SQuEELS',
    version=find_version("SQuEELS", "__init__.py"),
    description="Standard-based Quantification of EELS (SQuEELS) data.",
    long_description='A package for elemental quantification of EELS core loss edges using reference standard spectra.',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics',
    ],
    keywords=[
        'microscopy',
        'STEM',
        'TEM',
        'EELS',
    ],
    url='http://gitlab.com/rwebster/SQuEELS/',
    author='Robert Webster',
    author_email='rwhwebster@gmail.com',
    license='GPL v3',
    packages=['SQuEELS'],
    #package_data={'SQuEELS': []},
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'hyperspy',
        'pymc3',
        'matplotlib',
        'lmfit',
        'arviz',
        'tqdm',
    ],
    python_requires='>=3.6',
    #include_package_data=True,
    zip_safe=False,
    )