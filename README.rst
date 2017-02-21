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
- Documentation: http://pythonhosted.org/rrtoolbox/
- Download Page: https://pypi.python.org/pypi/rrtoolbox

Latest:
- Documentation: http://rrtools.readthedocs.io/en/latest/
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

Examples
========
Examples are found in the directory examples_ and unit tests in tests_.

Installation
============
``pip install rrtools`` should work for most users.

Detailed information can be found in `documentation/pyserial.rst`_.

The usual setup.py for Python_ libraries is used for the source distribution.
Windows installers are also available (see download link above).

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

Releases
========

All releases follow semantic rules proposed in https://www.python.org/dev/peps/pep-0440/ and http://semver.org/

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

A basic usage would be typing `./imrestore tests/im1* --lens` in the terminal which species
to imrestore to load from the test path images that start with im1 specified by the wildcard
'*' and the option --lens adds, as its name implies, lens to the retinal area.

* So, it loads this image which presents some flares and noise:
![Retina photo 1]
(https://github.com/davtoh/RRtools/blob/master/tests/im1_1.jpg)

* The second image is from a different perspective of the same retinal area but has information
that the other does not have:
![Retina photo 2]
(https://github.com/davtoh/RRtools/blob/master/tests/im1_2.jpg)

* And Voilà they are merged into one! notice how the flares tend to disappear and the lens
were added too! Because it is in development it still is not that pretty...
![Rethina photo result]
(https://github.com/davtoh/RRtools/blob/master/tests/_restored_im1_1.jpg)


Usage
=====
For help just type in './imrestore --help', it could not be easier than that! but a demo_
is available using the jupyter notebook to generate the desired commands to learn or use
in the console while still running the program.

>Contributions and bug reports are appreciated.
>author: David Toro
>e-mail: davsamirtor@gmail.com
>project: https://github.com/davtoh/RRtools