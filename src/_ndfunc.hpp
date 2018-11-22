// ############################################################################
// _ndfunc.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDFUNC_HPP
#define _NDFUNC_HPP

#include <complex>
#include <vector>

#include "_ndarray.hpp"
#include "_ndtype.hpp"

namespace nd {
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
    ndarray<T> ascontiguousarray(ndarray<T> const& x);

    /*
     * Create 1-D array from elements.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     1-D array containing elements from `x`.
     */
    template <typename T>
    ndarray<T> r_(std::vector<T> const& x);

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
    ndarray<T> arange(T const start, T const stop, T const step);

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
    ndarray<T> linspace(T const start, T const stop, size_t const num, bool const endpoint = true);

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
    ndarray<T> zeros(shape_t const& shape);

    /*
     * Returns
     * -------
     * out : ndarray<T>
     *     Array of ones of the given shape.
     */
    template <typename T>
    ndarray<T> ones(shape_t const& shape);

    /*
     * Returns
     * -------
     * out : ndarray<T>
     *     Array of `value` of the given shape.
     */
    template <typename T>
    ndarray<T> full(shape_t const& shape, T const value);

    /*
     * 2-D array with ones on a diagonal and zeros elsewhere.
     *
     * Parameters
     * ----------
     * N : size_t const
     *     Number of rows.
     * M : size_t const
     *     Number of columns.
     * k : int const
     *     Index of the diagonal.
     *     Positive values refer to upper-diagonals.
     *     Negative values refer to lower-diagonals.
     *
     * Returns
     * -------
     * I : ndarray<T>
     *     (N, M) array where all elements are 0, except on the k-th diagonal
     *     where they are 1.
     */
    template <typename T>
    ndarray<T> eye(size_t const N, size_t const M, int const k = 0);

    /*
     * Logical OR along specified axes.
     *
     * Parameters
     * ----------
     * x : ndarray<bool> const&
     *     N-D input array.
     * axes : std::vector<size_t> const&
     *     Subset of dimensions along which to reduce.
     * keepdims : bool const
     *     If true, the axes which are reduced are left in the result as
     *     dimensions with size 1.
     * out : ndarray<bool> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<bool>
     *     OR-reduced array with
     *     * (x.ndim() - axes.size()) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    ndarray<bool> any(ndarray<bool> const& x, std::vector<size_t> const& axes, bool const keepdims = false, ndarray<bool>* const out = nullptr);

    /*
     * Logical OR along all axes.
     *
     * Parameters
     * ----------
     * x : ndarray<bool> const&
     *     N-D input array.
     *
     * Returns
     * -------
     * y : ndarray<bool>
     *     (1,) OR-reduced array.
     */
    ndarray<bool> any(ndarray<bool> const& x);

    /*
     * Logical AND along specified axes.
     *
     * Parameters
     * ----------
     * x : ndarray<bool> const&
     *     N-D input array.
     * axes : std::vector<size_t> const&
     *     Subset of dimensions along which to reduce.
     * keepdims : bool const
     *     If true, the axes which are reduced are left in the result as
     *     dimensions with size 1.
     * out : ndarray<bool> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<bool>
     *     AND-reduced array with
     *     * (x.ndim() - axes.size()) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    ndarray<bool> all(ndarray<bool> const& x, std::vector<size_t> const& axes, bool const keepdims = false, ndarray<bool>* const out = nullptr);

    /*
     * Logical AND along all axes.
     *
     * Parameters
     * ----------
     * x : ndarray<bool> const&
     *     N-D input array.
     *
     * Returns
     * -------
     * y : ndarray<bool>
     *     (1,) AND-reduced array.
     */
    ndarray<bool> all(ndarray<bool> const& x);

    /*
     * Element-wise closeness check for floating point types.
     *
     * Parameters
     * ----------
     * out : ndarray<bool> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * close_enough: ndarray<bool>
     *     |x - y| <= (atol + rtol * |y|)
     */
    template <typename T>
    ndarray<bool> isclose(ndarray<T> const& x, ndarray<T> const& y, T const rtol = 1e-5, T const atol = 1e-8, ndarray<bool>* const out = nullptr);

    /*
     * Element-wise closeness check for floating point types.
     *
     * Returns
     * -------
     * all_close_enough: bool
     *     all(|x - y| <= (atol + rtol * |y|))
     */
    template <typename T>
    bool allclose(ndarray<T> const& x, ndarray<T> const& y, T const rtol = 1e-5, T const atol = 1e-8);

    /*
     * Sum of array elements over given axes.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Elements to sum.
     * axes : std::vector<size_t> const&
     *     Subset of dimensions along which to reduce.
     * keepdims : bool const
     *     If true, the axes which are reduced are left in the result as
     *     dimensions with size 1.
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     ADD-reduced array with
     *     * (x.ndim() - axes.size()) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> sum(ndarray<T> const& x, std::vector<size_t> const& axes, bool const keepdims = false, ndarray<T>* const out = nullptr);

    /*
     * Sum of array elements.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Elements to sum.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (1,) ADD-reduced array.
     */
    template <typename T>
    ndarray<T> sum(ndarray<T> const& x);

    /*
     * Product of array elements over given axes.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Elements to multiply.
     * axes : std::vector<size_t> const&
     *     Subset of dimensions along which to reduce.
     * keepdims : bool const
     *     If true, the axes which are reduced are left in the result as
     *     dimensions with size 1.
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     TIMES-reduced array with
     *     * (x.ndim() - axes.size()) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> prod(ndarray<T> const& x, std::vector<size_t> const& axes, bool const keepdims = false, ndarray<T>* const out = nullptr);

    /*
     * Product of array elements.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Elements to multiply.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (1,) TIMES-reduced array.
     */
    template <typename T>
    ndarray<T> prod(ndarray<T> const& x);

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
     * out : ndarray<T> * const
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
     * out : ndarray<T> * const
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
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     sin(x)
     */
    template <typename T>
    ndarray<T> sin(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric cosine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Angle [rad].
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     cos(x)
     */
    template <typename T>
    ndarray<T> cos(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric tangent.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Angle [rad].
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     tan(x)
     */
    template <typename T>
    ndarray<T> tan(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric inverse sine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Value in [-1, 1].
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arcsin(x) \in [-\pi/2, \pi/2]
     */
    template <typename T>
    ndarray<T> arcsin(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric inverse cosine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Value in [-1, 1].
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arccos(x) \in [0, \pi]
     */
    template <typename T>
    ndarray<T> arccos(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric inverse tangent.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arctan(x) \in [-\pi/2, \pi/2]
     */
    template <typename T>
    ndarray<T> arctan(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric inverse tangent of x1 / x2, choosing the quadrant correctly.
     *
     * Parameters
     * ----------
     * x1 : ndarray<T> const&
     * x2 : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arctan(x1 / x2) \in [-\pi, \pi]
     */
    template <typename T>
    ndarray<T> arctan2(ndarray<T> const& x1, ndarray<T> const& x2, ndarray<T>* const out = nullptr);

    /*
     * Convert angles from degrees to radians.
     *
     * Parameters
     * ----------
     * deg : ndarray<T> const&
     *     Angle [deg]
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * rad : ndarray<T>
     *     Angle [rad]
     */
    template <typename T>
    ndarray<T> deg2rad(ndarray<T> const& deg, ndarray<T>* const out = nullptr);

    /*
     * Convert angles from radians to degrees.
     *
     * Parameters
     * ----------
     * rad : ndarray<T> const&
     *     Angle [rad]
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * deg : ndarray<T>
     *     Angle [deg]
     */
    template <typename T>
    ndarray<T> rad2deg(ndarray<T> const& rad, ndarray<T>* const out = nullptr);

    /*
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     sin(\pi x) / (\pi x)
     */
    template <typename T>
    ndarray<T> sinc(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise exponentiatian.
     *
     * Parameters
     * ----------
     * base : ndarray<T> const&
     * exponent : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * res : ndarray<T>
     *     base ** exponent
     */
    template <typename T>
    ndarray<T> pow(ndarray<T> const& base, ndarray<T> const& exponent, ndarray<T>* const out = nullptr);

    /*
     * Element-wise base-E exponentiation.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     e ** x
     */
    template <typename T>
    ndarray<T> exp(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise base-2 exponentiation.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     2 ** x
     */
    template <typename T>
    ndarray<T> exp2(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise base-10 exponentiation.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     10 ** x
     */
    template <typename T>
    ndarray<T> exp10(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise base-E logarithm.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     ln(x)
     */
    template <typename T>
    ndarray<T> log(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise base-2 logarithm.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     log_{2}(x)
     */
    template <typename T>
    ndarray<T> log2(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise base-10 logarithm.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     log_{10}(x)
     */
    template <typename T>
    ndarray<T> log10(ndarray<T> const& x, ndarray<T>* const out = nullptr);

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
     * Standard deviation over given axes.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axes : std::vector<size_t> const&
     *     Subset of dimensions along which to compute the statistic.
     * keepdims : bool const
     *     If true, the axes which are reduced are left in the result as
     *     dimensions with size 1.
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     STD-reduced array with
     *     * (x.ndim() - axes.size()) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> std(ndarray<T> const& x, std::vector<size_t> const& axes, bool const keepdims = false, ndarray<T>* const out = nullptr);

    /*
     * Standard deviation of array elements.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (1,) STD-reduced array.
     */
    template <typename T>
    ndarray<T> std(ndarray<T> const& x);

    /*
     * Average over given axes.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axes : std::vector<size_t> const&
     *     Subset of dimensions along which to compute the statistic.
     * keepdims : bool const
     *     If true, the axes which are reduced are left in the result as
     *     dimensions with size 1.
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     MEAN-reduced array with
     *     * (x.ndim() - axes.size()) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> mean(ndarray<T> const& x, std::vector<size_t> const& axes, bool const keepdims = false, ndarray<T>* const out = nullptr);

    /*
     * Average of array elements.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (1,) MEAN-reduced array.
     */
    template <typename T>
    ndarray<T> mean(ndarray<T> const& x);

    /*
     * Range of values over given axes.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axes : std::vector<size_t> const&
     *     Subset of dimensions along which to compute the statistic.
     * keepdims : bool const
     *     If true, the axes which are reduced are left in the result as
     *     dimensions with size 1.
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (MAX - MIN)-reduced array with
     *     * (x.ndim() - axes.size()) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> ptp(ndarray<T> const& x, std::vector<size_t> const& axes, bool const keepdims = false, ndarray<T>* const out = nullptr);

    /*
     * Range of array elements.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (1,) (MAX - MIN)-reduced array.
     */
    template <typename T>
    ndarray<T> ptp(ndarray<T> const& x);

    /*
     * Minimum of array elements over given axes.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axes : std::vector<size_t> const&
     *     Subset of dimensions along which to reduce.
     * keepdims : bool const
     *     If true, the axes which are reduced are left in the result as
     *     dimensions with size 1.
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     MIN-reduced array with
     *     * (x.ndim() - axes.size()) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> min(ndarray<T> const& x, std::vector<size_t> const& axes, bool const keepdims = false, ndarray<T>* const out = nullptr);

    /*
     * Minimum of array elements.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (1,) MIN-reduced array.
     */
    template <typename T>
    ndarray<T> min(ndarray<T> const& x);

    /*
     * Maximum of array elements over given axes.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axes : std::vector<size_t> const&
     *     Subset of dimensions along which to reduce.
     * keepdims : bool const
     *     If true, the axes which are reduced are left in the result as
     *     dimensions with size 1.
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     MAX-reduced array with
     *     * (x.ndim() - axes.size()) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> max(ndarray<T> const& x, std::vector<size_t> const& axes, bool const keepdims = false, ndarray<T>* const out = nullptr);

    /*
     * Maximum of array elements.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (1,) MAX-reduced array.
     */
    template <typename T>
    ndarray<T> max(ndarray<T> const& x);

    /*
     * Sort an array.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to sort.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     Sorted copy of `x`.
     */
    template <typename T>
    ndarray<T> sort(ndarray<T> const& x, size_t const axis);

    /*
     * Indices that would sort an array.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to sort.
     *
     * Returns
     * -------
     * y : ndarray<size_t>
     *     Array of indices that sort `x` along dimension `axis`.
     */
    template <typename T>
    ndarray<size_t> argsort(ndarray<T> const& x, size_t const axis);

    /*
     * Element-wise round-up.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     */
    template <typename T>
    ndarray<T> ceil(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise round-down.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     */
    template <typename T>
    ndarray<T> floor(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Element-wise clip.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * down : T const
     * up : T const
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     Saturate `x` to lie in [down, up].
     */
    template <typename T>
    ndarray<T> clip(ndarray<T> const& x, T const down, T const up, ndarray<T>* const out = nullptr);

    /*
     * Element-wise indication number's sign.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     -1 if x < 0, 0 if x==0, 1 if x > 0.
     */
    template <typename T>
    ndarray<T> sign(ndarray<T> const& x, ndarray<T>* const out = nullptr);

    /*
     * Angle of complex argument.
     *
     * Parameters
     * ----------
     * x : ndarray<std::complex<T>> const&
     * out : ndarray<T> * const
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
}

#endif // _NDFUNC_HPP
