// ############################################################################
// _ndtype.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDTYPE_HPP
#define _NDTYPE_HPP

#include <type_traits>

#include "_ndforward.hpp"

namespace nd {
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
        return (std::is_same<T, cfloat>::value ||
                std::is_same<T, cdouble>::value ||
                std::is_same<T, cldouble>::value);
    }

    template <typename T>
    constexpr bool is_arithmetic() {
        return is_bool<T>() || is_int<T>() || is_float<T>() || is_complex<T>();
    }
}

#endif // _NDTYPE_HPP
