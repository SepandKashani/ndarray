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
#include <vector>

#include "Eigen/Eigen"

namespace nd {
    /*
     * Convenience types.
     */
    using cfloat = std::complex<float>;
    using cdouble = std::complex<double>;
    using cldouble = std::complex<long double>;
    using byte_t = uint8_t;



    /*
     * byte alignment: must be one of {2**k, k > 0}.
     *
     * Must set nd::byte_alignment to sizeof(<largest type you want to use>).
     */
    constexpr size_t byte_alignment = sizeof(cldouble);
    using shape_t  = std::vector<size_t>;
    using index_t = std::vector<size_t>;
    using stride_t = std::vector<int>;


    /*
     * Static type checks helpers.
     */
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



    /*
     * Interoperability types.
     *
     * These are used to interface with the Eigen3 library.
     */
    using mapS_t = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    template <typename T> using A_t = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    template <typename T> using M_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    template <typename T> using mapA_t = Eigen::Map<A_t<T>, Eigen::Unaligned, mapS_t>;
    template <typename T> using mapM_t = Eigen::Map<M_t<T>, Eigen::Unaligned, mapS_t>;
}

#endif // _NDTYPE_HPP
