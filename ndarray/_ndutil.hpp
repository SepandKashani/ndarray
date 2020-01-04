// ############################################################################
// _ndutil.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDUTIL_HPP
#define _NDUTIL_HPP

#include <algorithm>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "Eigen/Eigen"

#include "_ndforward.hpp"

namespace {
    template <typename T>
    std::ostream& operator<<(std::ostream& os,
                             std::vector<T> const& v) {
        os << "{";
        for(size_t i = 0, N = v.size(); i < N; ++i) {
            os << v[i] << ((i != N-1) ? ", " : "}");
        }
        return os;
    }

    template <typename T>
    std::ostream& operator<<(std::ostream& os,
                             nd::ndarray<T> const& x) {
        namespace ndu = nd::util;
        namespace ndui = nd::util::interop;

        auto x_ptr = const_cast<nd::ndarray<T>*>(&x);
        ndu::NDARRAY_ASSERT(ndui::is_eigen_mappable(x_ptr),
                            "Only 1d/2d arrays can be plotted.");
        auto ex = ndui::aseigenarray<T, nd::mapA_t<T>>(x_ptr, false);

        if(x.ndim() == 1) {
            Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]", "", "");
            os << ex->format(fmt);
        } else if (x.ndim() == 2) {
            Eigen::IOFormat fmt(Eigen::StreamPrecision, 0, ", ", "\n", "[", "]", "[", "]");
            os << ex->format(fmt);
        }

        return os;
    }
}

namespace nd::util {
    template <typename T>
    void NDARRAY_ASSERT(bool const cond,
                        T const& msg) {
        if(!cond) {
            throw std::runtime_error(msg);
        }
    }

    shape_t predict_shape_broadcast(shape_t const& lhs,
                                    shape_t const& rhs) {
        std::stringstream error_msg;
        error_msg << "Operands could not be broadcast together with shapes "
                  << "(" << lhs << ", " << rhs << ").\n";

        size_t const ndim_ohs = std::max<size_t>(lhs.size(), rhs.size());
        shape_t lhs_bcast(ndim_ohs, 1);
        shape_t rhs_bcast(ndim_ohs, 1);
        std::copy_n(lhs.rbegin(), lhs.size(), lhs_bcast.rbegin());
        std::copy_n(rhs.rbegin(), rhs.size(), rhs_bcast.rbegin());

        shape_t out(ndim_ohs, 1);
        for(size_t i = 0; i < out.size(); ++i) {
            size_t const& _lhs = lhs_bcast[i];
            size_t const& _rhs = rhs_bcast[i];
            size_t&       _out = out[i];

            NDARRAY_ASSERT((_lhs == _rhs) || (_lhs == 1) || (_rhs == 1),
                           error_msg.str());
            _out = std::max<size_t>(_lhs, _rhs);
        }

        return out;
    }

    shape_t predict_shape_reduction(shape_t const& shape,
                                    size_t const axis) {
        NDARRAY_ASSERT(axis < shape.size(), "Parameter[axis] is out of bounds.");

        shape_t out(shape);
        out[axis] = 1;

        return out;
    }

    slice::slice() {};

    slice::~slice() {};

    slice::slice(int const stop):
        m_start(0),
        m_stop(stop),
        m_step(1) {}

    slice::slice(int const start,
                 int const stop):
        m_start(start),
        m_stop(stop),
        m_step(1) {}

    slice::slice(int const start,
                 int const stop,
                 int const step):
        m_start(start),
        m_stop(stop),
        m_step(step) {
        NDARRAY_ASSERT(step != 0, "Zero steps are not allowed.");
    }

    int slice::start() const {
        return m_start;
    }

    int slice::stop() const {
        return m_stop;
    }

    int slice::step() const {
        return m_step;
    }

    slice slice::map_limits(size_t const length) const {
        if(((m_start > m_stop) && (m_step > 0)) ||
           ((m_stop > m_start) && (m_step < 0))) {
            return slice(0, 0, m_step);
        }

        if(m_step > 0) {
            return slice(std::max<int>(0, m_start),
                         std::min<int>(m_stop, length),
                         m_step);
        } else {
            return slice(std::min<int>(m_start, length - 1),
                         std::max<int>(-1, m_stop),
                         m_step);
        }
    }

    template <typename F, typename T1, typename T2>
    void apply(F ufunc,
               ndarray<T1>* const in,
               ndarray<T2>* const out) {
        NDARRAY_ASSERT(in != nullptr, "Parameter[in] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(out != nullptr, "Parameter[out] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(in->shape() == out->shape(), "Parameter[out] must have same dimensions as Parameter[in].");

        std::transform(in->begin(), in->end(), out->begin(), ufunc);
    }

    template <typename F, typename T1, typename T2>
    void apply(F ufunc,
               ndarray<T1>* const in_1,
               ndarray<T1>* const in_2,
               ndarray<T2>* const out) {
        NDARRAY_ASSERT(in_1 != nullptr, "Parameter[in_1] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(in_2 != nullptr, "Parameter[in_2] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(out != nullptr, "Parameter[out] must point to a valid ndarray instance.");

        auto const sh_out_correct = predict_shape_broadcast(in_1->shape(), in_2->shape());
        NDARRAY_ASSERT(out->shape() == sh_out_correct,
                       "Parameter[in_1, in_2] do not broadcast to Parameter[out] dimensions.");

        auto in_1_bcast = in_1->broadcast_to(sh_out_correct);
        auto in_2_bcast = in_2->broadcast_to(sh_out_correct);
        std::transform(in_1_bcast.begin(), in_1_bcast.end(),
                       in_2_bcast.begin(), out->begin(),
                       ufunc);
    }

    template <typename F, typename T>
    void reduce3D(F ufunc,
                  ndarray<T>* const in,
                  ndarray<T>* const out,
                  size_t const axis,
                  T const init) {
        NDARRAY_ASSERT(axis <= 2, "Parameter[axis] must be one of {0, 1, 2}.");

        NDARRAY_ASSERT(in != nullptr, "Parameter[in] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(in->ndim() == 3, "Parameter[in] must be 3D.");

        NDARRAY_ASSERT(out != nullptr, "Parameter[out] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(out->ndim() == 3, "Parameter[out] must be 3D.");
        for(size_t i = 0; i < in->ndim(); ++i) {
            if(i != axis) {
                NDARRAY_ASSERT(out->shape()[i] == in->shape()[i],
                               "Parameters[in, out] have incompatible dimensions.");
            } else {
                NDARRAY_ASSERT(out->shape()[i] == 1,
                               "Parameter[out] mush have length 1 along axis.");
            }
        }

        // For later optimizations.
        NDARRAY_ASSERT(in->is_contiguous(), "Parameter[in] must be contiguous.");
        NDARRAY_ASSERT(out->is_contiguous(), "Parameter[out] must be contiguous.");

        std::vector<size_t> idx_out(3); /* How to index the output. */
        std::vector<slice> select(3);   /* How to subsample the input. */
        std::vector<size_t> fix_axes;   /* Axes that are not reduced. */
        for(size_t i = 0; i < in->ndim(); ++i) {
            if(i != axis) {
                fix_axes.push_back(i);
            }
        }
        size_t const& size_lhs = in->shape()[fix_axes[0]];
        size_t const& size_rhs = in->shape()[fix_axes[1]];

        for(size_t i = 0; i < size_lhs; ++i) {
            for(size_t j = 0; j < size_rhs; ++j) {
                switch(axis) {
                    case 0:
                        select[0] = slice();
                        select[1] = slice(i, i + 1);
                        select[2] = slice(j, j + 1);
                        idx_out[0] = 0;
                        idx_out[1] = i;
                        idx_out[2] = j;
                        break;

                    case 1:
                        select[0] = slice(i, i + 1);
                        select[1] = slice();
                        select[2] = slice(j, j + 1);
                        idx_out[0] = i;
                        idx_out[1] = 0;
                        idx_out[2] = j;
                        break;

                    case 2:
                        select[0] = slice(i, i + 1);
                        select[1] = slice(j, j + 1);
                        select[2] = slice();
                        idx_out[0] = i;
                        idx_out[1] = j;
                        idx_out[2] = 0;
                        break;
                }

                auto sub_in = in->operator()(select);
                out->operator[](idx_out) =
                    std::accumulate(sub_in.begin(), sub_in.end(), init, ufunc);
            }
        }
    }

    template <typename F, typename T>
    void reduce(F ufunc,
                ndarray<T>* const in,
                ndarray<T>* const out,
                size_t const axis,
                T const init) {
        NDARRAY_ASSERT(in != nullptr, "Parameter[in] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(out != nullptr, "Parameter[out] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(in->ndim() == out->ndim(), "Parameters[in, out] must have same rank.");
        NDARRAY_ASSERT(axis < in->ndim(), "Parameter[axis] must be one of {0, ..., in->ndim() - 1}.");
        NDARRAY_ASSERT(out->is_contiguous(), "Parameter[out] must be contiguous.");
        for(size_t i = 0; i < in->ndim(); ++i) {
            if(i != axis) {
                NDARRAY_ASSERT(out->shape()[i] == in->shape()[i],
                               "Parameters[in, out] have incompatible dimensions.");
            } else {
                NDARRAY_ASSERT(out->shape()[i] == 1,
                               "Parameter[out] mush have length 1 along axis.");
            }
        }

        // 3D-equivalent axis/shape parameterization
        size_t axis3D;
        shape_t sh_in3D(3);
        shape_t sh_out3D(3);
        if(axis == 0) {
            // in(a, b, c) -> in3D(a, 1, b*c)
            // out(1, b, c) -> out3D(1, 1, b*c)
            axis3D = 0;
            sh_in3D[0] = in->shape()[0];
            sh_in3D[1] = 1;
            sh_in3D[2] = std::accumulate(in->shape().begin() + 1, in->shape().end(),
                                         1, std::multiplies<size_t>());
            sh_out3D[0] = 1;
            sh_out3D[1] = 1;
            sh_out3D[2] = sh_in3D[2];
        } else if(axis == in->ndim() - 1) {
            // in(a, b, c) -> in3D(a*b, 1, c)
            // out(a, b, 1) -> out3D(a*b, 1, 1)
            axis3D = 2;
            sh_in3D[0] = std::accumulate(in->shape().begin(), in->shape().end() - 1,
                                         1, std::multiplies<size_t>());
            sh_in3D[1] = 1;
            sh_in3D[2] = in->shape()[in->ndim() - 1];
            sh_out3D[0] = sh_in3D[0];
            sh_out3D[1] = 1;
            sh_out3D[2] = 1;
        } else {
            // in(a, b, c, d, e) -> in3D(a*b, c, d*e)
            // out(a, b, 1, d, e) -> out3D(a*b, 1, d*e)
            axis3D = 1;
            sh_in3D[0] = std::accumulate(in->shape().begin(), in->shape().begin() + axis,
                                         1, std::multiplies<size_t>());;
            sh_in3D[1] = in->shape()[axis];
            sh_in3D[2] = std::accumulate(in->shape().end() - (in->ndim() - 1 - axis), in->shape().end(),
                                         1, std::multiplies<size_t>());;
            sh_out3D[0] = sh_in3D[0];
            sh_out3D[1] = 1;
            sh_out3D[2] = sh_in3D[2];
        }

        auto in3D  = in->reshape(sh_in3D);
        auto out3D = out->reshape(sh_out3D);
        reduce3D(ufunc, &in3D, &out3D, axis3D, init);
        // At this point `out` contains the result.
    }
}

namespace nd::util::interop {
    template <typename T>
    bool is_eigen_mappable(ndarray<T>* const x) {
        NDARRAY_ASSERT(x != nullptr, "Parameter[x] must point to a valid ndarray instance.");

        auto overlap_check = [](stride_t::value_type const& stride) -> bool { return stride % sizeof(T) == 0; };
        bool const no_overlaps = std::all_of(x->strides().begin(), x->strides().end(), overlap_check);

        auto domain_check = [](stride_t::value_type const& stride) -> bool { return stride > 0; };
        bool const positive_strides = std::all_of(x->strides().begin(), x->strides().end(), domain_check);

        bool const is_1d = (x->ndim() == 1);
        bool const is_2d = (x->ndim() == 2);

        bool const is_mappable = (is_1d || is_2d) && no_overlaps && positive_strides;
        return is_mappable;
    }

    template <typename T, typename EigenFormat>
    std::unique_ptr<EigenFormat> aseigenarray(ndarray<T>* const x,
                                              bool const check_mappability) {
        constexpr bool ok_format = (std::is_same<EigenFormat, mapA_t<T>>::value ||
                                    std::is_same<EigenFormat, mapM_t<T>>::value);
        static_assert(ok_format, "Only {nd::mapA_t<T>, nd::mapM_t<T>} types are supported.");

        NDARRAY_ASSERT(x != nullptr, "Parameter[x] must point to a valid ndarray instance.");

        if(check_mappability) {
            NDARRAY_ASSERT(is_eigen_mappable(x), "Parameter[x] is not Eigen-mappable.");
        }

        size_t N_dim0(0), N_dim1(0);
        int N_stride_0(0), N_stride_1(0);
        if(x->ndim() == 1) {
            N_dim0 = 1;
            N_dim1 = x->shape()[0];
            N_stride_0 = 0;
            N_stride_1 = x->strides()[0] / sizeof(T);
        } else {
            N_dim0 = x->shape()[0];
            N_dim1 = x->shape()[1];
            N_stride_0 = x->strides()[0] / sizeof(T);
            N_stride_1 = x->strides()[1] / sizeof(T);
        }

        nd::mapS_t strides(N_stride_0, N_stride_1);
        auto map = std::make_unique<EigenFormat>(x->data(), N_dim0, N_dim1, strides);
        return map;
    }

    template <typename Derived>
    ndarray<typename Derived::Scalar> asndarray(Eigen::ArrayBase<Derived> const& x) {
        size_t const N_rows = x.rows();
        size_t const N_cols = x.cols();
        shape_t const shape({N_rows, N_cols});

        using T = typename Derived::Scalar;
        ndarray<T> y(shape);
        auto ey = aseigenarray<T, mapA_t<T>>(&y, false);
        (*ey) = x;

        return y;
    }
}

#endif // _NDUTIL_HPP
