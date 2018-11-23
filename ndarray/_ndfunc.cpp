// ############################################################################
// _ndfunc.cpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _FUNC_CPP
#define _FUNC_CPP

#include <algorithm>
#include <vector>

#include "_ndarray.hpp"
#include "_ndfunc.hpp"
#include "_ndtype.hpp"
#include "_ndutil.hpp"

namespace nd {
    template <typename T>
    inline ndarray<T> ascontiguousarray(ndarray<T> const& x) {
        if(x.is_contiguous()) {
            return x;
        } else {
            return x.copy();
        }
    }

    template <typename T>
    inline ndarray<T> r_(std::vector<T> const& x) {
        ndarray<T> y(shape_t({x.size()}));

        std::copy_n(x.cbegin(), x.size(), y.data());
        return y;
    }

    template <typename T>
    inline ndarray<T> zeros(shape_t const& shape) {
        ndarray<T> out(shape);
        std::fill_n(out.data(), out.size(), T(0.0));

        return out;
    }

    template <typename T>
    inline ndarray<T> ones(shape_t const& shape) {
        ndarray<T> out(shape);
        std::fill_n(out.data(), out.size(), T(1.0));

        return out;
    }

    template <typename T>
    inline ndarray<T> full(shape_t const& shape, T const value) {
        ndarray<T> out(shape);
        std::fill_n(out.data(), out.size(), value);

        return out;
    }

    template <typename T>
    inline ndarray<T> eye(size_t const N, size_t const M, int const k) {
        util::NDARRAY_ASSERT(N == M, "Only square arrays currently supported.");
        util::NDARRAY_ASSERT(k == 0, "Only main diagonal currently supported.");

        ndarray<T> I = zeros<T>({N, M});
        int const step = I.strides()[0] + I.strides()[1];
        byte_t* data = reinterpret_cast<byte_t*>(I.data());
        for(size_t i = 0; i < N; ++i) {
            reinterpret_cast<T*>(data)[0] = static_cast<T>(1.0);
            data += step;
        }

        return I;
    }
}

#endif // _NDFUNC_CPP
