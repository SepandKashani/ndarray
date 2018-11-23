// ############################################################################
// _ndlinalg.cpp
// =============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDLINALG_CPP
#define _NDLINALG_CPP

#include "_ndarray.hpp"
#include "_ndlinalg.hpp"

namespace nd { namespace linalg {
    template <typename T>
    inline ndarray<T> mm(ndarray<T> const& A, ndarray<T> const& B, ndarray<T>* const out) {
        return A + B;
    }
}}

#endif // _NDLINALG_CPP
