=========================================================
RRtools - Raster Restoration Tools  |build-status| |docs|
=========================================================

Overview
========

This module encapsulates algorithms for the restoration of images and it is
specialized in retinal images.

RRtoolbox is a python package which contains source code designed to process images built
mainly using OpenCV.

RRtoolFC is a development tool using sequential function charts (FC stands for Function Chart)

Imrestore is an application to restore images in general but in this case is configured to
restore retinal images. Because it is still in development it is the alpha program for testing
and to let the users find out about its utilities. Do not hesitate to share it to the world,
let everyone know how awesome it is!! (be warned that it is for research purposes)

Stable:

    - Documentation: http://pythonhosted.org/rrtools
    - Download Page: https://pypi.python.org/pypi/rrtoolbox

Latest:

    - Documentation: http://rrtools.readthedocs.io/
    - Project Homepage: https://github.com/davtoh/RRtools

BSD license, (C) 2015-2017 David Toro <davsamirtor@gmail.com>

Documentation
=============

For API documentation, usage and examples see files in the "documentation"
directory.  The ".rst" files can be read in any text editor or being converted to
HTML or PDF using Sphinx_. A HTML version is online at
http://rrtools.readthedocs.io/en/latest/

Read RRtoolbox (Retinal Restauration Toolbox) manual_ or the dissertation_
which made this project possible with all the concepts.

Examples are found in the directory examples_ and unit tests in tests_.

Installation
============
``pip install rrtools`` should work for most users.

The usual setup.py for Python_ libraries are used for the source distribution.
But OpenCV must be installed separately usually compiled from source. However
RRtoolbox has a mock module for cv2 called cv2_mock to let the user use the
functions that do not need OpenCV. Notice that this imports RRtoolbox.cv2_mock
as cv2.

To install OpenCV without much hassle I recommend installing the binaries from
the `Unofficial Windows Binaries for Python`_ and for Debian distributions I
provide the bash `OpenCV linux installation`_ so that the user can compile
openCV (it can take some time). Bear in mind that for Linux it downloads the
latest 2.9 version instead of the new OpenCV version 3 because it does not
consent to using non-free sources. So you must accept the terms for OpenCV 2.9.

Once rrtools is successfully installed you can import the toolbox in python as:

    >>>> import RRtoolbox as rr

Releases
========

All releases follow semantic rules proposed in https://www.python.org/dev/peps/pep-0440/ 
and http://semver.org/

To create executable from source code::

    $ pyinstaller -p ./ -n imrestore -F ./imrestore.py/ --version-file=version


Testing and application
=======================

To test imrestore script::

    $ python imrestore.py tests/im1*

To test imrestore executable::

    $./dist/imrestore tests/im1*

To test imrestore executable if in dist folder::

    $ ./imrestore ../tests/im1*

A basic usage would be typing ``./imrestore tests/im1* --lens`` in the terminal which species
to imrestore to load from the test path images that start with im1 specified by the wildcard
'*' and the option ``--lens`` adds, as its name implies, lens to the retinal area.

* So, it loads this image which presents some flares and noise:

.. figure:: https://github.com/davtoh/RRtools/blob/master/tests/im1_1.jpg:align: center
    :scale: 10%

    Retina photo 1

* The second image is from a different perspective of the same retinal area but has information 
  that the other does not have:

.. figure:: https://github.com/davtoh/RRtools/blob/master/tests/im1_2.jpg
    :align: center
    :scale: 10%

    Retina photo 2

* And Voilà they are merged into one! notice how the flares tend to disappear and the lens were 
  added too! Because it is in development it still is not that pretty...

.. figure:: https://github.com/davtoh/RRtools/blob/master/tests/_restored_im1_1.jpg
    :align: center
    :scale: 10%

    Rethina photo result


Usage
=====

For help just type in ``./imrestore --help``, it could not be easier than that! but a demo_
is available using the jupyter notebook to generate the desired commands to learn or use
in the console while still running the program.

- Contributions and bug reports are appreciated.
- author: David Toro
- e-mail: davsamirtor@gmail.com
- project: https://github.com/davtoh/RRtools

.. _`documentation/index.rst`: https://github.com/davtoh/RRtools/blob/master/documentation/index.rst
.. _examples: https://github.com/davtoh/RRtools/tree/master/examples
.. _tests: https://github.com/davtoh/RRtools/tree/master/tests
.. _Python: http://python.org/
.. _Sphinx: http://sphinx-doc.org/
.. _pyinstaller: http://www.pyinstaller.org/
.. |build-status| image:: https://travis-ci.org/pyserial/pyserial.svg?branch=master
   :target: https://github.com/davtoh/RRtools/releases
   :alt: Build status
.. |docs| image:: https://readthedocs.org/projects/pyserial/badge/?version=latest
   :target: http://rrtools.readthedocs.io/
   :alt: Documentation
.. _manual: https://github.com/davtoh/RRtools/blob/master/documentation/_build/latex/RRtoolbox.pdf
.. _dissertation:
.. _demo: https://github.com/davtoh/RRtools/blob/master/ImRestore_demo.ipynb
.. _`Unofficial Windows Binaries for Python`: http://www.lfd.uci.edu/~gohlke/pythonlibs/
.. _`OpenCV linux installation`: https://github.com/davtoh/RRtools/blob/master/install_opencv.sh
