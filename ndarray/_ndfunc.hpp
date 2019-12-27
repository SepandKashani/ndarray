// ############################################################################
// _ndfunc.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDFUNC_HPP
#define _NDFUNC_HPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <functional>
#include <type_traits>
#include <vector>

#include "_ndarray.hpp"
#include "_ndtype.hpp"
#include "_ndutil.hpp"

namespace nd {
    template <typename T> class ndarray;

    /*
     * Returns
     * -------
     * pi : T
     *     Mathematical constant \pi \approx 3.1415
     */
    template <typename T>
    T constexpr pi() {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        return static_cast<T>(M_PI);
    }

    /*
     * Returns
     * -------
     * e : T
     *     Euler's constant e \approx 2.71828
     */
    template <typename T>
    T constexpr e() {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        return static_cast<T>(M_E);
    }

    /*
     * Returns
     * -------
     * j : T
     *     Imaginary constant j = \sqrt(-1)
     */
    template <typename T>
    T constexpr j() {
        static_assert(is_complex<T>(), "Only {complex} types allowed.");

        return T(0.0, 1.0);
    }

    /*
     * Return a contiguous array in memory (C-order).
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Input array.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     Contiguous array of same shape and content as `x`.
     *     If `x` was already contiguous, then a view is returned.
     */
    template <typename T>
    ndarray<T> ascontiguousarray(ndarray<T> const& x) {
        if(x.is_contiguous()) {
            return x;
        } else {
            return x.copy();
        }
    }

    /*
     * Create 1-D array from elements.
     *
     * Parameters
     * ----------
     * x : std::vector<T> const&
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     1-D array containing elements from `x`.
     */
    template <typename T>
    ndarray<T> r_(std::vector<T> const& x) {
        ndarray<T> y(shape_t({x.size()}));

        std::copy_n(x.cbegin(), x.size(), y.data());
        return y;
    }

    /*
     * Evenly-spaced values in a given interval.
     *
     * Parameters
     * ----------
     * start : T const
     *     Start of the interval.
     *     The interval includes this value.
     * stop : T const
     *     End of interval.
     *     The interval does not include this value, except in some cases where
     *     step is not an integer and floating point round-off affects the length of out.
     * step : T const
     *     Spacing between values.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     1-D array of evenly-spaced values.
     *     For T=<floats>, the length of the result is `ceil((stop - start)/step)`.
     *     Because of floating point overflow, this rule may result in the last
     *     element being greater than `stop`.
     */
    template <typename T>
    ndarray<T> arange(T const start, T const stop, T const step) {
        static_assert(is_signed_int<T>() || is_float<T>(),
                      "Only {signed_int, float} types allowed." );
        util::NDARRAY_ASSERT(std::abs(step) > 0, "Parameter[step] cannot be 0.");
        util::NDARRAY_ASSERT(((start <= stop) && (step > 0)) || ((start >= stop) && (step < 0)),
                             "Impossible start/stop/step combination provided.");

        size_t N = std::ceil((stop - start) / step);
        ndarray<T> y(shape_t({N}));

        y.data()[0] = start;
        for(size_t i = 1; i < N; ++i) {
            y.data()[i] = y.data()[i - 1] + step;
        }

        return y;
    }

    /*
     * Evenly spaced numbers over a specified interval.
     *
     * Parameters
     * ----------
     * start : T const
     *     Start of the interval.
     * stop : T const
     *     End of the interval.
     * N : size_t const
     *     Number of samples to generate.
     * endpoint : bool const
     *     If true (default), `stop` is the last sample.
     *     If false, `stop` is not included.
     *
     * Returns
     * -------
     * samples : ndarray<T>
     *     (N,) array with equally-spaced values.
     *     If `endpoint` is true,  samples[i] = start + i * ((stop - start) / (N - 1)).
     *     If `endpoint` is false, samples[i] = start + i * ((stop - start) / N).
     */
    template <typename T>
    ndarray<T> linspace(T const start, T const stop, size_t const N, bool const endpoint = true) {
        static_assert(is_float<T>(), "Only {float} types allowed.");
        util::NDARRAY_ASSERT(N > 1, "Parameter[N] must be at least 2.");

        ndarray<T> y(shape_t({N}));
        T step = (stop - start) / static_cast<T>(endpoint ? (N - 1) : N);

        y.data()[0] = start;
        for(size_t i = 1; i < N - 1; ++i) {
            y.data()[i] = y.data()[i - 1] + step;
        }
        // Last element computed outside loop to avoid overflowing past `stop` with FP arithmetic.
        y.data()[N - 1] = stop - static_cast<T>(endpoint ? 0 : ((stop - start) / static_cast<T>(N)));

        return y;
    }

    /*
     * Coordinate arrays from coordinate vectors.
     *
     * Parameters
     * ----------
     * x : std::vector<ndarray<T>> const&
     *     1-D arrays representing coordinates of a grid.
     * sparse : bool const
     *     Return grids that are not fleshed-out to conserve memory.
     *
     * Returns
     * -------
     * mesh : std::vector<ndarray<T>>
     *     Given x[i].size() = s[i], return (s[0], s[1], ..., s[N-1]) shaped
     *     arrays where mesh[i][{a, ..., z}] = x[i][{<corresponding letter from {a, ..., z}>}].
     */
    template <typename T>
    std::vector<ndarray<T>> meshgrid(std::vector<ndarray<T>> const& x, bool const sparse = false);

    /*
     * Returns
     * -------
     * out : ndarray<T>
     *     Array of zeros of the given shape.
     */
    template <typename T>
    ndarray<T> zeros(shape_t const& shape) {
        ndarray<T> out(shape);
        std::fill_n(out.data(), out.size(), static_cast<T>(0.0));

        return out;
    }

    /*
     * Returns
     * -------
     * out : ndarray<T>
     *     Array of ones of the given shape.
     */
    template <typename T>
    ndarray<T> ones(shape_t const& shape) {
        ndarray<T> out(shape);
        std::fill_n(out.data(), out.size(), static_cast<T>(1.0));

        return out;
    }

    /*
     * Returns
     * -------
     * out : ndarray<T>
     *     Array of `value` of the given shape.
     */
    template <typename T>
    ndarray<T> full(shape_t const& shape, T const value) {
        ndarray<T> out(shape);
        std::fill_n(out.data(), out.size(), value);

        return out;
    }

    /*
     * 2-D array with ones on the main diagonal and zeros elsewhere.
     *
     * Parameters
     * ----------
     * N : size_t const
     *     Number of rows.
     *
     * Returns
     * -------
     * I : ndarray<T>
     *     (N, N) array where all elements are 0, except on the main diagonal where they are 1.
     */
    template <typename T>
    ndarray<T> eye(size_t const N) {
        ndarray<T> I = zeros<T>({N, N});
        int const step = I.strides()[0] + I.strides()[1];
        byte_t* data = reinterpret_cast<byte_t*>(I.data());
        for(size_t i = 0; i < N; ++i) {
            reinterpret_cast<T*>(data)[0] = static_cast<T>(1.0);
            data += step;
        }

        return I;
    }

    /*
     * Logical OR along specified axis.
     *
     * Parameters
     * ----------
     * x : ndarray<bool> const&
     *     N-D input array.
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as dimension of size 1.
     * out : ndarray<bool>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<bool>
     *     OR-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T, typename BOOL = bool>
    ndarray<BOOL> any(ndarray<T> const& x,
                      size_t const axis,
                      bool const keepdims = false,
                      ndarray<T>* const out = nullptr) {
        /*
         * Why not use `ndarray<bool>` directly as return type?
         * Doing so for some reason gives the following error message:
         *     error: return type ‘class nd::ndarray<bool>’ is incomplete
         *
         * Don't know how to solve the issue using forward-declaration.
         */
        static_assert(is_bool<T>(), "Only {bool} types allowed.");

        if(out == nullptr) {
            shape_t const& shape_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(shape_y);
            util::reduce(std::logical_or<T>(), const_cast<ndarray<T>*>(&x), &y, axis, false);
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(std::logical_or<T>(), const_cast<ndarray<T>*>(&x), out, axis, false);
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    /*
     * Logical AND along specified axis.
     *
     * Parameters
     * ----------
     * x : ndarray<bool> const&
     *     N-D input array.
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<bool>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<bool>
     *     AND-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T, typename BOOL = bool>
    ndarray<BOOL> all(ndarray<T> const& x,
                      size_t const axis,
                      bool const keepdims = false,
                      ndarray<T>* const out = nullptr) {
        /*
         * Why not use `ndarray<bool>` directly as return type?
         * Doing so for some reason gives the following error message:
         *     error: return type ‘class nd::ndarray<bool>’ is incomplete
         *
         * Don't know how to solve the issue using forward-declaration.
         */
        static_assert(is_bool<T>(), "Only {bool} types allowed.");

        if(out == nullptr) {
            shape_t const& shape_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(shape_y);
            util::reduce(std::logical_and<T>(), const_cast<ndarray<T>*>(&x), &y, axis, true);
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(std::logical_and<T>(), const_cast<ndarray<T>*>(&x), out, axis, true);
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    /*
     * Element-wise closeness check for floating point types.
     *
     * Parameters
     * ----------
     * out : ndarray<bool>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * close_enough: ndarray<bool>
     *     |x - y| <= (atol + rtol * |y|)
     */
    template <typename T, typename BOOL = bool>
    ndarray<BOOL> isclose(ndarray<T> const& x,
                          ndarray<T> const& y,
                          ndarray<bool>* const out = nullptr,
                          double const rtol = 1e-5,
                          double const atol = 1e-8) {
        /*
         * Why not use `ndarray<bool>` directly as return type?
         * Doing so for some reason gives the following error message:
         *     error: return type ‘class nd::ndarray<bool>’ is incomplete
         *
         * Don't know how to solve the issue using forward-declaration.
         */
        static_assert(is_float<T>() || is_complex<T>() , "Only {float, complex} types allowed.");

        auto ufunc = [atol, rtol](T const& _x, T const& _y) -> bool {
            auto lhs = std::abs(_x - _y);
            auto rhs = atol + rtol * std::abs(_y);
            return lhs <= rhs;
        };
        if(out == nullptr) {
            shape_t const& shape_out = util::predict_shape_broadcast(x.shape(), y.shape());
            ndarray<bool> close_enough(shape_out);

            util::apply(ufunc,
                        const_cast<ndarray<T>*>(&x),
                        const_cast<ndarray<T>*>(&y),
                        &close_enough);
            return close_enough;
        } else {
            util::apply(ufunc,
                        const_cast<ndarray<T>*>(&x),
                        const_cast<ndarray<T>*>(&y),
                        out);
            return *out;
        }
    }

    /*
     * Element-wise closeness check for floating point types.
     *
     * Returns
     * -------
     * all_close_enough: bool
     *     all(|x - y| <= (atol + rtol * |y|))
     */
    template <typename T, typename BOOL = bool>
    BOOL allclose(ndarray<T> const& x,
                  ndarray<T> const& y,
                  double const rtol = 1e-5,
                  double const atol = 1e-8) {
        /*
         * Why not use `bool` directly as return type?
         * Doing so for some reason gives the following error message:
         *     error: return type ‘class nd::ndarray<bool>’ is incomplete
         *
         * Don't know how to solve the issue using forward-declaration.
         */
        static_assert(is_float<T>() || is_complex<T>() , "Only {float, complex} types allowed.");

        ndarray<BOOL> closeness = isclose(x, y, nullptr, rtol, atol).ravel();
        return all(closeness, 0, true).data()[0];
    }

    /*
     * Sum of array elements over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Elements to sum.
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     ADD-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> sum(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims = false,
                   ndarray<T>* const out = nullptr) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        if(out == nullptr) {
            shape_t const& shape_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(shape_y);
            util::reduce(std::plus<T>(), const_cast<ndarray<T>*>(&x), &y, axis, static_cast<T>(0.0));
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(std::plus<T>(), const_cast<ndarray<T>*>(&x), out, axis, static_cast<T>(0.0));
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    /*
     * Product of array elements over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Elements to multiply.
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     TIMES-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> prod(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims = false,
                   ndarray<T>* const out = nullptr) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        if(out == nullptr) {
            shape_t const& shape_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(shape_y);
            util::reduce(std::multiplies<T>(), const_cast<ndarray<T>*>(&x), &y, axis, static_cast<T>(1.0));
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(std::multiplies<T>(), const_cast<ndarray<T>*>(&x), out, axis, static_cast<T>(1.0));
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    /*
     * Join a sequence of arrays along an existing axis.
     *
     * Parameters
     * ----------
     * x : std::vector<ndarray<T> const&> const&
     *     Arrays to join.
     *     All arrays must have the same shape, except along the dimension specified by `axis`.
     * axis : size_t const
     *     Dimension along which arrays are joined.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     */
    template <typename T>
    ndarray<T> concatenate(std::vector<ndarray<T> const&> const& x, size_t const axis, ndarray<T>* const out = nullptr);

    /*
     * Join a sequence of arrays along a new axis.
     *
     * Parameters
     * ----------
     * x : std::vector<ndarray<T> const&> const&
     *     Arrays to join.
     *     All arrays must have the same shape.
     * axis : size_t const
     *     Dimension in the output array along which arrays are joined.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     */
    template <typename T>
    ndarray<T> stack(std::vector<ndarray<T> const&> const& x, size_t const axis, ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric sine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Angle [rad].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     sin(x)
     */
    template <typename T>
    ndarray<T> sin(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::sin(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise trigonometric cosine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Angle [rad].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     cos(x)
     */
    template <typename T>
    ndarray<T> cos(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::cos(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise trigonometric tangent.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Angle [rad].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     tan(x)
     */
    template <typename T>
    ndarray<T> tan(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::tan(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise trigonometric inverse sine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Value in [-1, 1].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arcsin(x) \in [-\pi/2, \pi/2], NaN if out-of-range.
     */
    template <typename T>
    ndarray<T> arcsin(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::asin(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise trigonometric inverse cosine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Value in [-1, 1].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arccos(x) \in [0, \pi], NaN if out-of-range.
     */
    template <typename T>
    ndarray<T> arccos(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::acos(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise trigonometric inverse tangent.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arctan(x) \in [-\pi/2, \pi/2]
     */
    template <typename T>
    ndarray<T> arctan(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::atan(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise trigonometric inverse tangent of x1 / x2, choosing the quadrant correctly.
     *
     * Parameters
     * ----------
     * x1 : ndarray<T> const&
     * x2 : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arctan(x1 / x2) \in [-\pi, \pi]
     */
    template <typename T>
    ndarray<T> arctan2(ndarray<T> const& x1, ndarray<T> const& x2, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x1, T const& _x2) -> T { return std::atan2(_x1, _x2); };
        if(out == nullptr) {
            ndarray<T> y(util::predict_shape_broadcast(x1.shape(), x2.shape()));
            util::apply(ufunc, const_cast<ndarray<T>*>(&x1), const_cast<ndarray<T>*>(&x2), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x1), const_cast<ndarray<T>*>(&x2), out);
            return *out;
        }
    }

    /*
     * Convert angles from degrees to radians.
     *
     * Parameters
     * ----------
     * deg : ndarray<T> const&
     *     Angle [deg]
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * rad : ndarray<T>
     *     Angle [rad]
     */
    template <typename T>
    ndarray<T> deg2rad(ndarray<T> const& deg, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        T constexpr ratio = static_cast<T>(M_PI / 180.0);
        auto ufunc = [ratio](T const& _deg) -> T { return _deg * ratio; };
        if(out == nullptr) {
            ndarray<T> rad(deg.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&deg), &rad);
            return rad;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&deg), out);
            return *out;
        }
    }

    /*
     * Convert angles from radians to degrees.
     *
     * Parameters
     * ----------
     * rad : ndarray<T> const&
     *     Angle [rad]
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * deg : ndarray<T>
     *     Angle [deg]
     */
    template <typename T>
    ndarray<T> rad2deg(ndarray<T> const& rad, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        T constexpr ratio = static_cast<T>(180.0 / M_PI);
        auto ufunc = [ratio](T const& _rad) -> T { return _rad * ratio; };
        if(out == nullptr) {
            ndarray<T> deg(rad.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&rad), &deg);
            return deg;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&rad), out);
            return *out;
        }
    }

    /*
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     sin(\pi x) / (\pi x)
     */
    template <typename T>
    ndarray<T> sinc(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T {
            T const px = pi<T>() * ((_x != 0) ? _x : 1e-20);
            return std::sin(px) / px;
        };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise absolution value.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     abs(x)
     */
    template <typename T>
    ndarray<T> abs(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_signed_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {signed_int, float, complex} types supported.");

        auto ufunc = [](T const& _x) -> T { return std::abs(_x); };
        if(out == nullptr) {
            ndarray<T> _out(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &_out);
            return _out;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise base-E exponentiation.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     e ** x
     */
    template <typename T>
    ndarray<T> exp(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>() || is_complex<T>(),
                      "Only {float, complex} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::exp(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise base-E logarithm.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     ln(x) \in \bR, NaN if x < 0.
     */
    template <typename T>
    ndarray<T> log(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::log(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Find the unique elements of an array.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Input array.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (N,) array with sorted unique values of `x.ravel()`.
     */
    template <typename T>
    ndarray<T> unique(ndarray<T> const& x);

    /*
     * Standard deviation over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     STD-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     *
     * Notes
     * -----
     * STD is implemented as mean(abs(x - x.mean()) ** 2).
     */
    template <typename T>
    ndarray<T> std(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims = false,
                   ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>() || is_complex<T>(),
                      "Only {float, complex} types allowed.");

        auto n = x - mean(x, axis, true);
        abs(n, &n);
        n *= n;
        auto y = mean(n, axis, keepdims, out);
        auto sqrt = [](T const& _var) -> T { return std::sqrt(_var); };
        util::apply(sqrt, &y, &y);

        return y;
    }

    /*
     * Average over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     MEAN-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> mean(ndarray<T> const& x,
                    size_t const axis,
                    bool const keepdims = false,
                    ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>() || is_complex<T>(),
                      "Only {float, complex} types allowed.");

        ndarray<T> y = sum(x, axis, true, out);
        T const N = static_cast<T>(x.shape()[axis]);
        y /= N;

        return (keepdims ? y : y.squeeze({axis}));
    }

    /*
     * Minimum of array elements over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     MIN-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> min(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims = false,
                   ndarray<T>* const out = nullptr) {
        static_assert(is_int<T>() || is_float<T>(),
                      "Only {int, float} types allowed.");

        auto f_min = [](T const& _x, T const& _y) -> T const& { return std::min(_x, _y); };
        T constexpr init = std::numeric_limits<T>::max();

        if(out == nullptr) {
            shape_t const& shape_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(shape_y);
            util::reduce(f_min, const_cast<ndarray<T>*>(&x), &y, axis, init);
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(f_min, const_cast<ndarray<T>*>(&x), out, axis, init);
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    /*
     * Maximum of array elements over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     MAX-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> max(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims = false,
                   ndarray<T>* const out = nullptr) {
        static_assert(is_int<T>() || is_float<T>(),
                      "Only {int, float} types allowed.");

        auto f_max = [](T const& _x, T const& _y) -> T const& { return std::max(_x, _y); };
        T constexpr init = std::numeric_limits<T>::lowest();

        if(out == nullptr) {
            shape_t const& shape_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(shape_y);
            util::reduce(f_max, const_cast<ndarray<T>*>(&x), &y, axis, init);
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(f_max, const_cast<ndarray<T>*>(&x), out, axis, init);
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    /*
     * Element-wise ceiling.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     */
    template <typename T>
    ndarray<T> ceil(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::ceil(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise floor.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     */
    template <typename T>
    ndarray<T> floor(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_float<T>(), "Only {float} types allowed.");

        auto ufunc = [](T const& _x) -> T { return std::floor(_x); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise clip.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * down : T const
     * up : T const
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     Saturate `x` to lie in [down, up].
     */
    template <typename T>
    ndarray<T> clip(ndarray<T> const& x, T const down, T const up, ndarray<T>* const out = nullptr) {
        static_assert(is_int<T>() || is_float<T>(), "Only {int, float} types allowed.");
        util::NDARRAY_ASSERT(down <= up, "Parameter[down] must be \\le Parameter[up].");

        auto ufunc = [down, up](T const& _x) -> T { return std::clamp<T>(_x, down, up); };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Element-wise indication of number's sign.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     -1 if x < 0, 0 if x==0, 1 if x > 0.
     */
    template <typename T>
    ndarray<T> sign(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_signed_int<T>() || is_float<T>(),
                      "Only {signed_int, float} types allowed.");

        T constexpr p = static_cast<T>(1.0);
        T constexpr m = static_cast<T>(-1.0);
        T constexpr z = static_cast<T>(0.0);
        auto ufunc = [p, m, z](T const& _x) -> T {
            if(_x > z) {
                return p;
            } else if(_x < z) {
                return m;
            } else {
                return z;
            }
        };
        if(out == nullptr) {
            ndarray<T> y(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &y);
            return y;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }

    /*
     * Angle of complex argument.
     *
     * Parameters
     * ----------
     * x : ndarray<std::complex<T>> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * arg : ndarray<T>
     *     Counter-clockwise angle [rad] from the positive real axis.
     */
    template <typename T>
    ndarray<T> angle(ndarray<std::complex<T>> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise real-part extraction.
     *
     * Parameters
     * ----------
     * x : ndarray<std::complex<T>> const&
     *
     * Returns
     * -------
     * out : ndarray<T>
     *     View on real-part of `x`.
     */
    template <typename T>
    ndarray<T> real(ndarray<std::complex<T>> const& x) {
        static_assert(is_float<T>(), "Only {complex} types are supported.");

        stride_t float_strides(x.ndim() + 1, sizeof(T));
        std::copy_n(x.strides().begin(), x.ndim(), float_strides.begin());

        shape_t float_shape(x.ndim() + 1, 2);
        std::copy_n(x.shape().begin(), x.ndim(), float_shape.begin());

        byte_t* const float_data = reinterpret_cast<byte_t*>(x.data());
        ndarray<T> float_x(x.base(), float_data, float_shape, float_strides);

        std::vector<util::slice> selection(float_x.ndim(), util::slice());
        selection[float_x.ndim() - 1] = util::slice(0, 1);
        ndarray<T> out = float_x(selection).squeeze({float_x.ndim() - 1});
        return out;
    }

    /*
     * Element-wise imag-part extraction.
     *
     * Parameters
     * ----------
     * x : ndarray<std::complex<T>> const&
     *
     * Returns
     * -------
     * out : ndarray<T>
     *     View on imag-part of `x`.
     */
    template <typename T>
    ndarray<T> imag(ndarray<std::complex<T>> const& x) {
        static_assert(is_float<T>(), "Only {complex} types are supported.");

        stride_t float_strides(x.ndim() + 1, sizeof(T));
        std::copy_n(x.strides().begin(), x.ndim(), float_strides.begin());

        shape_t float_shape(x.ndim() + 1, 2);
        std::copy_n(x.shape().begin(), x.ndim(), float_shape.begin());

        byte_t* const float_data = reinterpret_cast<byte_t*>(x.data());
        ndarray<T> float_x(x.base(), float_data, float_shape, float_strides);

        std::vector<util::slice> selection(float_x.ndim(), util::slice());
        selection[float_x.ndim() - 1] = util::slice(1, 2);
        ndarray<T> out = float_x(selection).squeeze({float_x.ndim() - 1});
        return out;
    }

    /*
     * Element-wise conjugation.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * out : ndarray<T>
     *     conj(x)
     */
    template <typename T>
    ndarray<T> conj(ndarray<T> const& x, ndarray<T>* const out = nullptr) {
        static_assert(is_complex<T>(), "Only {complex} types supported.");

        auto ufunc = [](T const& _x) -> T { return std::conj(_x); };
        if(out == nullptr) {
            ndarray<T> _out(x.shape());
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), &_out);
            return _out;
        } else {
            util::apply(ufunc, const_cast<ndarray<T>*>(&x), out);
            return *out;
        }
    }
}

#endif // _NDFUNC_HPP
