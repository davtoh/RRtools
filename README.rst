=========================================================
RRtools - Raster Restoration Tools  |build-status| |docs|
=========================================================

Overview
========

This module encapsulates algorithms for the restoration of images and it
specializes on retinal images. The following names are used to distinguish projects and algorithms
separately:

* RRtoolbox is a python package which contains source code designed to process images built
mainly using OpenCV. It could thought as the core of the project.

* RRtoolFC is a development tool using sequential function charts (FC stands for Function Chart).
This tool is still under development and can be thought as the programming interface.

* Imrestore is an application to restore images in general but in this case is configured to
restore retinal images. Because it is still under development it is currently used for testing
and to let the users find out about its utilities.

Do not hesitate to share or use the project. Currently the project is under development and it is for
research purposes.

Stable:

    - Documentation: http://pythonhosted.org/rrtools
    - Download Page: https://pypi.python.org/pypi/rrtoolbox

Latest:

    - Documentation: http://rrtools.readthedocs.io/
    - Project Homepage: https://github.com/davtoh/RRtools

BSD license, David Toro <davsamirtor@gmail.com>

Documentation
=============

For API documentation, usage and examples see files in the "documentation"
directory.  The ".rst" files can be read in any text editor or be converted to
HTML or PDF using Sphinx_. A HTML version is available online at
http://rrtools.readthedocs.io/en/latest/

Read RRtoolbox (Retinal Restauration Toolbox) manual_ or the dissertation_
which made this project possible with all the concepts and explanations.

Examples are found in the directory examples_ and unit tests in tests_.

Installation
============
``pip install rrtools`` should work for most users.

The usual setup.py for Python_ libraries is used for the source distribution.
But in most cases OpenCV must be installed separately usually compiled from source. However
RRtoolbox has a mock module for cv2 called cv2_mock to let the user use the
functions that do not need OpenCV. Notice that this imports RRtoolbox.cv2_mock
as cv2.

To install OpenCV on windows without much hassle I recommend installing the binaries from
the `Unofficial Windows Binaries for Python`_ and for Debian distributions I
provide the bash `OpenCV linux installation`_ so that the user can compile
openCV (it can take some time). Bear in mind that for Linux it downloads the
latest 2.9 version instead of the new OpenCV version 3 because it does not
consent to using non-free sources. You must accept the terms for OpenCV 2.9.

Once rrtools is successfully installed you can import the toolbox in python as:

    >>>> import RRtoolbox as rr

Releases
========

All releases follow semantic rules proposed in https://www.python.org/dev/peps/pep-0440/ 
and http://semver.org/

Executable
==========

To create an executable from source code::

    $ pyinstaller -p ./ -n imrestore -F ./imrestore.py/ --version-file=version


Usage and testing
=================

Modern Frontier Vision CLI:
---------------------------

RRtools provides a modernized command-line interface for multi-modal scene restoration:

1. **Multi-Angle 360° Spherical Panoramic Stitcher**:
   Projects wide indoor/outdoor perspectives into equirectangular space with multi-band Laplacian pyramid seam blending, eliminating planar homography tearing and exposure seams::

       $ python -m RRtoolbox panorama assets/demo/panorama_input_1.jpg assets/demo/panorama_input_2.jpg --projection spherical --blend multiband -o panorama_restored.jpg

   Key Options:
     * ``--projection {spherical,cylindrical,planar}``: Camera remapping model (default: ``spherical``).
     * ``--blend {multiband,feather,linear,none}``: Seam blending mode (default: ``multiband``).
     * ``--fov <degrees>``: Horizontal Field of View (default: ``65.0``).
     * ``--levels <int>``: Laplacian pyramid octave levels for frequency blending (default: ``4``).

2. **Multi-Frame Burst Super-Resolution (MFSR) & SISR**:
   Sub-pixel ECC registration and Lanczos-4 accumulation (+45.4% optical sharpness gain)::

       $ python -m RRtoolbox super-res tests/im5_1.jpg tests/im5_2.jpg tests/im5_3.jpg --scale 2 -o super_restored.jpg

3. **Hardware & GPU Compute Diagnostics**:
   Probes physical GPU capabilities (CUDA, ONNX Runtime, OpenCV DNN) with graceful CPU fallback::

       $ python -m RRtoolbox hardware

Python API Usage:
-----------------
.. code-block:: python

    from RRtoolbox.geometry import PanoramicStitcher

    stitcher = PanoramicStitcher(projection="spherical", blend_mode="multiband")
    result = stitcher.stitch(["assets/demo/panorama_input_1.jpg", "assets/demo/panorama_input_2.jpg"])
    result.save("panorama_restored.jpg")
    print(f"Stitched {result.output_shape} in {result.elapsed_ms:.1f}ms with {result.inlier_count} inliers")

Classical Retinal Restoration (imrestore):
------------------------------------------

To test imrestore script::

    $ python imrestore.py tests/im1*

To test imrestore executable::

    $./dist/imrestore tests/im1*

To test imrestore executable if in dist folder::

    $ ./imrestore ../tests/im1*

A basic usage would be typing ``./imrestore tests/im1* --lens`` in the terminal which tells
 imrestore to load images from the test path that start with im1 specified by the wildcard
'*' and the option ``--lens`` which adds, as its name implies, lens to the retinal area.

* This will load the following image which presents some flares and noise:

.. figure:: https://github.com/davtoh/RRtools/blob/master/tests/im1_1.jpg
    :align: center
    :scale: 10%

    Retina photo 1

* The second image is from a different perspective of the same retinal area but has information 
  that the other does not have:

.. figure:: https://github.com/davtoh/RRtools/blob/master/tests/im1_2.jpg
    :align: center
    :scale: 10%

    Retina photo 2

* And Voilà they are merged into one. Notice how the flares disappear, hidden vascular details
  are reconstructed, and the simulated lens aperture is added!

.. figure:: https://github.com/davtoh/RRtools/blob/master/assets/demo/retina_restored.jpg?raw=true
    :align: center
    :scale: 10%

    Retina photo result (Restored with glare suppression and lens simulation)


Demos and Visual Guides
=======================

* **Showcase & Parameter Studio Guide**: See `DEMO.md`_ for full before/after comparisons,
  panoramic indoor stitching, and the interactive command generator.
* **Interactive Parameter Studio**: Open ``ImRestore_demo.html`` in your browser or run
  ``jupyter notebook ImRestore_demo.ipynb`` to visually tune parameters and automatically
  generate CLI commands.
* **Algorithmic Steps Walkthrough**: Explore ``RRtools_steps.ipynb`` for a step-by-step
  diagnostic breakdown of each phase of the restoration pipeline.


Help
====

For help just type in ``./imrestore --help``, it could not be easier than that! A demo_
is available using the jupyter notebook to generate the desired commands to learn or use
in the console while still running the program. Feel free to email me if you need help with
the code or would like to have additional features.

- Contributions and bug reports are appreciated.
- author: David Toro
- e-mail: davsamirtor@gmail.com
- project: https://github.com/davtoh/RRtools

.. _`DEMO.md`: https://github.com/davtoh/RRtools/blob/master/DEMO.md
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
.. _dissertation: https://github.com/davtoh/RRtools/blob/master/dissertation.pdf
.. _demo: https://github.com/davtoh/RRtools/blob/master/ImRestore_demo.ipynb
.. _`Unofficial Windows Binaries for Python`: http://www.lfd.uci.edu/~gohlke/pythonlibs/
.. _`OpenCV linux installation`: https://github.com/davtoh/RRtools/blob/master/install_opencv.sh
