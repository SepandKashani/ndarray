// ############################################################################
// _ndtype.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

/*
 * Project-wide types and constants.
 */

#ifndef _NDTYPE_HPP
#define _NDTYPE_HPP

#include <complex>

namespace nd {
    using byte_t = uint8_t;
    /*
     * byte alignment: must be one of {2**k, k > 0}.
     *
     * Must set nd::byte_alignment to sizeof(<largest type you want to use>).
     */
    constexpr size_t byte_alignment = sizeof(std::complex<long double>);
    using shape_t  = std::vector<size_t>;
    using stride_t = std::vector<int>;
}

#endif // _NDTYPE_HPP
