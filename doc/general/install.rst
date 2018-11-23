.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################

Installation
============

*Ndarray* is a header-only C++14 library and tested on x86_64 systems running Linux.

The following libraries must be available on the system before installing *Ndarray*:

+-------------+------------+
| Library     |    Version |
+=============+============+
| Eigen       |      3.3.5 |
+-------------+------------+
| FFTW        |      3.3.8 |
+-------------+------------+
| GTEST       |      1.8.1 |
+-------------+------------+
| Intel MKL   |   2018.0.3 |
+-------------+------------+

Headers
-------
::

    $ cd <root_dir>/
    $ mkdir -p build/ndarray
    $ cd build/ndarray
    $ cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON ../..
    $ make install


Documentation
-------------
::

    $ cd <root_dir>/
    $ mkdir -p build/doc
    $ sphinx-build -b html doc build/doc

Remarks
-------

* *Ndarray* is tested internally with GCC 8.2.1.
