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

package = 'RRtoolbox'
version = find_version(package, '__init__.py')
print("Package '{}' is in version {}".format(package,version))

packages = find_packages(exclude=['RRtoolFC*', 'thesis*', 'tests*'])
print("Packages to include {}".format(packages))

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

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
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Descriptors :: Memoization',
    ],
    keywords='sample setuptools development',
    platforms='any',
    install_requires=[#'opencv-python>=2.4.11,<3', # for image manipulation
                        'numpy>=1.9.2', # for array manipulation and Memory-mapped file
                        'dill>=0.2.4', # for serializing manipulation
                        'joblib>=0.8.4', # for memoization and Memory-mapped file
                        'pycallgraph>=1.0.1',
                        'pyqtgraph>=0.9.10', # for image array visualization and visual interfaces (it got pyqt in it)
                        'sympy>=1.0',
                        'matplotlib>=1.4.3',
                        'pyperclip>=1.5.26',
                      ],
    #scripts=['imrestore.py'],
    entry_points={
        'console_scripts': [
            'imrestore = RRtools.imrestore:shell',
        ],
    }
)
