# setup.py for rrtools
#
# Direct install (all systems):
#   "python setup.py install"
#
# For Python 3.x use the corresponding Python executable,
# e.g. "python3 setup.py ..."
#
# (C) 2015-2017 David Toro <davsamirtor@gmail.com>
#
import sys
import io
import os
import re

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages


def read(*names, **kwargs):
    """Python 2 and Python 3 compatible text file reading.
    Required for single-sourcing the version string.
    """
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    """
    Search the file for a version string.
    file_path contain string path components.
    Reads the supplied Python module as text without importing it.
    """
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# information of package
print(sys.version) # version of python

package = 'RRtoolbox'
version = find_version(package, '__init__.py')
print("Package '{}' is in version {}".format(package,version)) # package version

packages = find_packages(exclude=['RRtoolFC*', 'thesis*', 'tests*'])
print("Packages to include {}".format(packages)) # package includes

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
long_description = read(os.path.join(here, 'README.rst'))

setup(
    name="rrtools",
    description="Raster Restoration tools",
    version=version,
    author="David Toro",
    author_email="davsamirtor@gmail.com",
    url="https://github.com/davtoh/RRtools",
    packages=packages,
    #packages=['RRtoolbox', 'RRtoolbox.tools', 'RRtoolbox.lib'],
    license="BSD",
    long_description=long_description,
    # see complete list https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        #'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='development, images, restoration',
    platforms='any',
    install_requires=[
        'packaging>=20.0',
        'appdirs>=1.4',
        'numpy>=1.20',
        'scipy>=1.8',
        'dill>=0.3',
        'joblib>=1.0',
        'sympy>=1.8',
        'matplotlib>=3.5',
        'pyperclip>=1.8',
        'opencv-contrib-python>=4.5',
    ],
    scripts=['imrestore.py','expert.py', 'fastplt.py'],
    entry_points={
        'console_scripts': [
            'imrestore = RRtools.imrestore:shell',
        ],
    }
)
