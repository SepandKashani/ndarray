.. ############################################################################
.. install.rst
.. ===========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################

Installation
============

*Ndarray* is a header-only C++17 library tested on x86_64 systems running Linux. Dependencies are
installed via the `Conan <https://docs.conan.io/en/latest/index.html>`_ package manager.


Headers
-------
::

    $ BUILD_DIR="<root_dir>/build/ndarray"
    $ mkdir -p "${BUILD_DIR}" && cd "${BUILD_DIR}"
    $ conan install --build=missing ../..
    $ cmake -DCMAKE_BUILD_TYPE=[Debug, Release] ../..
    $ [sudo] make install
    $ "${BUILD_DIR}/bin/test_ndarray"


Documentation
-------------
::

    $ BUILD_DIR="<root_dir>/build/doc"
    $ mkdir -p "${BUILD_DIR}"
    $ sphinx-build -b html doc "${BUILD_DIR}"


Remarks
-------

* *Ndarray* is tested internally with GCC 9.2.0.
