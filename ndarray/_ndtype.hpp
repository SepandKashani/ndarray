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
#include <type_traits>

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

    template <typename T>
    constexpr bool is_bool() {
        return std::is_same<T, bool>::value;
    }

    template <typename T>
    constexpr bool is_int() {
        return std::is_integral<T>::value && (!is_bool<T>());
    }

    template <typename T>
    constexpr bool is_signed_int() {
        return std::is_signed<T>::value;
    }

    template <typename T>
    constexpr bool is_float() {
        return std::is_floating_point<T>::value;
    }

    template <typename T>
    constexpr bool is_complex() {
        return (std::is_same<T, std::complex<float>>::value ||
                std::is_same<T, std::complex<double>>::value ||
                std::is_same<T, std::complex<long double>>::value);
    }

    template <typename T>
    constexpr bool is_arithmetic() {
        return is_bool<T>() || is_int<T>() || is_float<T>() || is_complex<T>();
    }
}

#endif // _NDTYPE_HPP
