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
#include <initializer_list>
#include <set>
#include <sstream>
#include <vector>

#include "_ndforward.hpp"

namespace nd {
    template <typename T>
    T constexpr pi() {
        static_assert(is_float<T>(), "Only {float} types allowed.");
        return static_cast<T>(M_PI);
    }

    template <typename T>
    T constexpr e() {
        static_assert(is_float<T>(), "Only {float} types allowed.");
        return static_cast<T>(M_E);
    }

    template <typename T>
    T constexpr j() {
        static_assert(is_complex<T>(), "Only {complex} types allowed.");
        return T(0.0, 1.0);
    }

    template <typename T>
    ndarray<T> ascontiguousarray(ndarray<T> const& x) {
        return x.is_contiguous() ? x : x.copy();
    }

    template <typename T>
    ndarray<T> r_(std::initializer_list<T> ilist) {
        util::NDARRAY_ASSERT(ilist.size() > 0, "r_() requires at least one element.");

        ndarray<T> y({ilist.size()});
        std::copy_n(ilist.begin(), ilist.size(), y.data());
        return y;
    }

    template <typename T>
    ndarray<T> arange(T const start,
                      T const stop,
                      T const step) {
        static_assert(is_signed_int<T>() || is_float<T>(),
                      "Only {signed_int, float} types allowed." );
        util::NDARRAY_ASSERT(std::abs(step) > 0, "Parameter[step] cannot be 0.");
        util::NDARRAY_ASSERT(((start <= stop) && (step > 0)) ||
                             ((start >= stop) && (step < 0)),
                             "Impossible start/stop/step combination provided.");

        size_t N = std::ceil((stop - start) / step);
        ndarray<T> y({N});

        y.data()[0] = start;
        for(size_t i = 1; i < N; ++i) {
            y.data()[i] = y.data()[i - 1] + step;
        }

        return y;
    }

    template <typename T>
    ndarray<T> linspace(T const start,
                        T const stop,
                        size_t const N,
                        bool const endpoint) {
        static_assert(is_float<T>(), "Only {float} types allowed.");
        util::NDARRAY_ASSERT(N > 1, "Parameter[N] must be at least 2.");

        ndarray<T> y({N});
        T step = (stop - start) / static_cast<T>(endpoint ? (N - 1) : N);

        y.data()[0] = start;
        for(size_t i = 1; i < N - 1; ++i) {
            y.data()[i] = y.data()[i - 1] + step;
        }
        // Last element computed outside loop to avoid overflowing past `stop` with FP arithmetic.
        y.data()[N - 1] = stop - static_cast<T>(endpoint ? 0 : ((stop - start) / static_cast<T>(N)));

        return y;
    }

    template <typename T>
    std::vector<ndarray<T>> meshgrid(std::vector<ndarray<T>> const& x) {
        static_assert(is_arithmetic<T>(), "Only {bool, int, float, complex} types allowed.");

        for(auto const& _x : x) {
            util::NDARRAY_ASSERT(_x.ndim() == 1u, "Only 1d coordinate arrays allowed.");
        }

        std::vector<ndarray<T>> mesh;
        for(size_t i = 0, N = x.size(); i < N; ++i) {
            auto const& _x = x[i];

            shape_t sh(N, 1);
            sh[i] = _x.size();

            mesh.push_back(_x.copy().reshape(sh));
        }

        return mesh;
    }

    template <typename T>
    ndarray<T> full(shape_t const& shape,
                    T const value) {
        ndarray<T> out(shape);
        std::fill_n(out.data(), out.size(), value);

        return out;
    }

    template <typename T>
    ndarray<T> zeros(shape_t const& shape) {
        return full(shape, static_cast<T>(0.0));
    }

    template <typename T>
    ndarray<T> ones(shape_t const& shape) {
        return full<T>(shape, static_cast<T>(1.0));
    }

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

    template <typename T, typename BOOL>
    ndarray<BOOL> any(ndarray<T> const& x,
                      size_t const axis,
                      bool const keepdims,
                      ndarray<T>* const out) {
        /*
         * Why not use `ndarray<bool>` directly as return type?
         * Doing so for some reason gives the following error message:
         *     error: return type ‘class nd::ndarray<bool>’ is incomplete
         *
         * Don't know how to solve the issue using forward-declaration.
         */
        static_assert(is_bool<T>(), "Only {bool} types allowed.");

        if(out == nullptr) {
            auto const& sh_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(sh_y);
            util::reduce(std::logical_or<T>(), const_cast<ndarray<T>*>(&x), &y, axis, false);
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(std::logical_or<T>(), const_cast<ndarray<T>*>(&x), out, axis, false);
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    template <typename T, typename BOOL>
    ndarray<BOOL> all(ndarray<T> const& x,
                      size_t const axis,
                      bool const keepdims,
                      ndarray<T>* const out) {
        /*
         * Why not use `ndarray<bool>` directly as return type?
         * Doing so for some reason gives the following error message:
         *     error: return type ‘class nd::ndarray<bool>’ is incomplete
         *
         * Don't know how to solve the issue using forward-declaration.
         */
        static_assert(is_bool<T>(), "Only {bool} types allowed.");

        if(out == nullptr) {
            auto const& sh_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(sh_y);
            util::reduce(std::logical_and<T>(), const_cast<ndarray<T>*>(&x), &y, axis, true);
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(std::logical_and<T>(), const_cast<ndarray<T>*>(&x), out, axis, true);
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    template <typename T, typename BOOL>
    ndarray<BOOL> isclose(ndarray<T> const& x,
                          ndarray<T> const& y,
                          ndarray<bool>* const out,
                          double const rtol,
                          double const atol) {
        /*
         * Why not use `ndarray<bool>` directly as return type?
         * Doing so for some reason gives the following error message:
         *     error: return type ‘class nd::ndarray<bool>’ is incomplete
         *
         * Don't know how to solve the issue using forward-declaration.
         */
        static_assert(is_arithmetic<T>() , "Only {bool, int, float, complex} types allowed.");

        auto ufunc = [atol, rtol](T const& _x, T const& _y) -> bool {
            if constexpr (is_float<T>() || is_complex<T>()) {
                auto lhs = std::abs(_x - _y);
                auto rhs = atol + rtol * std::abs(_y);
                return lhs <= rhs;
            } else {
                // {bool, int} case
                return _x == _y;
            }
        };
        if(out == nullptr) {
            auto const& sh_out = util::predict_shape_broadcast(x.shape(), y.shape());
            ndarray<bool> close_enough(sh_out);

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

    template <typename T, typename BOOL>
    BOOL allclose(ndarray<T> const& x,
                  ndarray<T> const& y,
                  double const rtol,
                  double const atol) {
        /*
         * Why not use `bool` directly as return type?
         * Doing so for some reason gives the following error message:
         *     error: return type ‘class nd::ndarray<bool>’ is incomplete
         *
         * Don't know how to solve the issue using forward-declaration.
         */
        static_assert(is_arithmetic<T>() , "Only {bool, int, float, complex} types allowed.");

        ndarray<BOOL> closeness = isclose(x, y, nullptr, rtol, atol).ravel();
        return all(closeness, 0, true).data()[0];
    }

    template <typename T>
    ndarray<T> sum(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims,
                   ndarray<T>* const out) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        if(out == nullptr) {
            auto const& sh_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(sh_y);
            util::reduce(std::plus<T>(), const_cast<ndarray<T>*>(&x), &y, axis, static_cast<T>(0.0));
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(std::plus<T>(), const_cast<ndarray<T>*>(&x), out, axis, static_cast<T>(0.0));
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    template <typename T>
    ndarray<T> prod(ndarray<T> const& x,
                    size_t const axis,
                    bool const keepdims,
                    ndarray<T>* const out) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        if(out == nullptr) {
            auto const& sh_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(sh_y);
            util::reduce(std::multiplies<T>(), const_cast<ndarray<T>*>(&x), &y, axis, static_cast<T>(1.0));
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(std::multiplies<T>(), const_cast<ndarray<T>*>(&x), out, axis, static_cast<T>(1.0));
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    template <typename T>
    ndarray<T> stack(std::vector<ndarray<T>> const& x,
                     size_t const axis,
                     ndarray<T>* const out) {
        static_assert(is_arithmetic<T>(), "Only {bool, int, float, complex} types allowed.");

        {   // Validate shapes
            util::NDARRAY_ASSERT(x.size() > 0, "No array(s) to stack.");

            std::stringstream error_msg;
            error_msg << "Cannot concatenate arrays of shape {";
            for(size_t i = 0, N = x.size(); i < N; ++i) {
                error_msg << x[i].shape() << ((i != N-1) ? ", " : "");
            }
            error_msg << "}. \n";

            auto same_shape = [&x](ndarray<T> const& _x) -> bool {
                auto const& sh_1 = x[0].shape();
                auto const& sh_2 = _x.shape();
                return sh_1 == sh_2;
            };
            util::NDARRAY_ASSERT(std::all_of(x.begin(), x.end(), same_shape), error_msg.str());
        }

        {   // Validate axis
            size_t const N_axis = x[0].ndim();
            util::NDARRAY_ASSERT(axis <= N_axis, "Parameter[axis] is out of bounds.");
        }

        auto sh_y = x[0].shape();
        sh_y.insert(sh_y.begin() + axis, x.size());
        ndarray<T> y((out == nullptr) ? sh_y : *out);

        auto sh_x = sh_y;
        sh_x[axis] = 1;

        auto selection = std::vector(sh_y.size(), util::slice());
        for(int i = 0, N = x.size(); i < N; ++i) {
            selection[axis] = util::slice(i, i + 1);
            y(selection) = x[i].reshape(sh_x);
        }

        return y;
    }

    template <typename T>
    ndarray<T> sin(ndarray<T> const& x,
                   ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> cos(ndarray<T> const& x,
                   ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> tan(ndarray<T> const& x,
                   ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> arcsin(ndarray<T> const& x,
                      ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> arccos(ndarray<T> const& x,
                      ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> arctan(ndarray<T> const& x,
                      ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> arctan2(ndarray<T> const& x1,
                       ndarray<T> const& x2,
                       ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> deg2rad(ndarray<T> const& deg,
                       ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> rad2deg(ndarray<T> const& rad,
                       ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> sinc(ndarray<T> const& x,
                    ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> abs(ndarray<T> const& x,
                   ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> exp(ndarray<T> const& x,
                   ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> log(ndarray<T> const& x,
                   ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> unique(ndarray<T> const& x) {
        static_assert(is_bool<T>() || is_int<T>() || is_float<T>(),
                      "Only {bool, int, float} types allowed.");

        std::set<T> _unique(x.begin(), x.end());
        size_t const N = _unique.size();

        ndarray<T> y({N});
        std::copy(_unique.cbegin(), _unique.cend(), y.begin());
        return y;
    }

    template <typename T>
    ndarray<T> std(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims,
                   ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> mean(ndarray<T> const& x,
                    size_t const axis,
                    bool const keepdims,
                    ndarray<T>* const out) {
        static_assert(is_float<T>() || is_complex<T>(),
                      "Only {float, complex} types allowed.");

        auto y = sum(x, axis, true, out);
        T const N = static_cast<T>(x.shape()[axis]);
        y /= N;

        return (keepdims ? y : y.squeeze({axis}));
    }

    template <typename T>
    ndarray<T> min(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims,
                   ndarray<T>* const out) {
        static_assert(is_int<T>() || is_float<T>(),
                      "Only {int, float} types allowed.");

        auto f_min = [](T const& _x, T const& _y) -> T const& { return std::min(_x, _y); };
        T constexpr init = std::numeric_limits<T>::max();

        if(out == nullptr) {
            auto const& sh_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(sh_y);
            util::reduce(f_min, const_cast<ndarray<T>*>(&x), &y, axis, init);
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(f_min, const_cast<ndarray<T>*>(&x), out, axis, init);
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    template <typename T>
    ndarray<T> max(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims,
                   ndarray<T>* const out) {
        static_assert(is_int<T>() || is_float<T>(),
                      "Only {int, float} types allowed.");

        auto f_max = [](T const& _x, T const& _y) -> T const& { return std::max(_x, _y); };
        T constexpr init = std::numeric_limits<T>::lowest();

        if(out == nullptr) {
            auto const& sh_y = util::predict_shape_reduction(x.shape(), axis);
            ndarray<T> y(sh_y);
            util::reduce(f_max, const_cast<ndarray<T>*>(&x), &y, axis, init);
            return (keepdims ? y : y.squeeze({axis}));
        } else {
            util::reduce(f_max, const_cast<ndarray<T>*>(&x), out, axis, init);
            return (keepdims ? *out : out->squeeze({axis}));
        }
    }

    template <typename T>
    ndarray<T> ceil(ndarray<T> const& x,
                    ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> floor(ndarray<T> const& x,
                     ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> clip(ndarray<T> const& x,
                    T const down,
                    T const up,
                    ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> sign(ndarray<T> const& x,
                    ndarray<T>* const out) {
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

    template <typename T>
    ndarray<T> asfloat(ndarray<std::complex<T>> const& x) {
        static_assert(is_float<T>(), "Only {complex} types are supported.");

        stride_t str_f(x.ndim() + 1, sizeof(T));
        std::copy_n(x.strides().begin(), x.ndim(), str_f.begin());

        shape_t sh_f(x.ndim() + 1, 2);
        std::copy_n(x.shape().begin(), x.ndim(), sh_f.begin());

        byte_t* const data_f = reinterpret_cast<byte_t*>(x.data());
        ndarray<T> x_f(x.base(), data_f, sh_f, str_f);
        return x_f;
    }

    template <typename T>
    ndarray<T> real(ndarray<std::complex<T>> const& x) {
        static_assert(is_float<T>(), "Only {complex} types are supported.");

        auto x_f = asfloat(x);
        std::vector selection(x_f.ndim(), util::slice());
        selection[x_f.ndim() - 1] = util::slice(0, 1);
        auto out = x_f(selection).squeeze({x_f.ndim() - 1});
        return out;
    }

    template <typename T>
    ndarray<T> imag(ndarray<std::complex<T>> const& x) {
        static_assert(is_float<T>(), "Only {complex} types are supported.");

        auto x_f = asfloat(x);
        std::vector selection(x_f.ndim(), util::slice());
        selection[x_f.ndim() - 1] = util::slice(1, 2);
        auto out = x_f(selection).squeeze({x_f.ndim() - 1});
        return out;
    }

    template <typename T>
    ndarray<T> conj(ndarray<T> const& x,
                    ndarray<T>* const out) {
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
