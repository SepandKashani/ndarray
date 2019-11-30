// ############################################################################
// _ndutil.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDUTIL_HPP
#define _NDUTIL_HPP

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "Eigen/Eigen"

#include "_ndtype.hpp"

namespace nd { template <typename T> class ndarray; }

namespace {
    template <typename T> std::ostream& operator<<(std::ostream& os, std::vector<T> const& v);
    template <typename T> std::ostream& operator<<(std::ostream& os, nd::ndarray<T> const& x);
}

namespace nd::util {
    /*
     * assert()-like statement that does not deactivate in Release mode.
     *
     * Parameters
     * ----------
     * cond : bool const
     * msg : T const&
     */
    template <typename T>
    void NDARRAY_ASSERT(bool const cond, T const& msg) {
        if(!cond) {
            throw std::runtime_error(msg);
        }
    }

    /*
     * Determine output dimensions based on broadcasting rules of binary operators.
     *
     * Parameters
     * ----------
     * lhs : shape_t const&
     *     Shape of left operand.
     * rhs : shape_t const&
     *     Shape of right operand.
     *
     * Result
     * ------
     * out : shape_t
     *     Shape of result.
     */
    inline shape_t predict_shape_broadcast(shape_t const& lhs, shape_t const& rhs) {
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

    /*
     * Determine output dimensions based on reduction specification.
     *
     * Parameters
     * ----------
     * shape : shape_t const&
     *     Shape of input.
     * axis : size_t const
     *     Dimension along which a reduction is made.
     *
     * Returns
     * -------
     * out : shape_t
     *     Shape of result.
     */
    inline shape_t predict_shape_reduction(shape_t const& shape, size_t const axis) {
        NDARRAY_ASSERT(axis < shape.size(), "Parameter[axis] is out of bounds.");

        shape_t out = shape;
        out[axis] = 1;

        return out;
    }

    /*
     * NumPy-like slice object.
     */
    class slice {
        private:
            int m_start      = 0;
            int m_stop       = std::numeric_limits<int>::max();
            int m_step       = 1;

        public:
            slice() {};

            /*
             * Parameters
             * ----------
             * stop : int const
             *     Termination index (exclusive).
             */
            slice(int const stop):
                m_start(0),
                m_stop(stop),
                m_step(1) {}

            /*
             * Parameters
             * ----------
             * start : int const
             *     Initialisation index (inclusive).
             * stop : int const
             *     Termination index (exclusive).
             */
            slice(int const start, int const stop):
                m_start(start),
                m_stop(stop),
                m_step(1) {}

            /*
             * Parameters
             * ----------
             * start : int const
             *     Initialisation index (inclusive).
             * stop : int const
             *     Termination index (exclusive). `-1` is a valid termination index to include the
             *     first element of an array.
             * step : int const
             *     Step size (non-zero).
             */
            slice(int const start, int const stop, int const step):
                m_start(start),
                m_stop(stop),
                m_step(step) {
                    NDARRAY_ASSERT(step != 0, "Zero steps are not allowed.");
                }

            ~slice() {};

            int start() const { return m_start; }

            int stop() const { return m_stop; }

            int step() const { return m_step; }

            /*
             * Set (ambiguous) slice limits to correct values for an
             * array dimension of length `length`.
             *
             * Parameters
             * ----------
             * length : size_t const
             *     Length of the dimension to which the slice is applied.
             *
             * Returns
             * -------
             * compact_slice : slice
             *     New slice object with start/stop values correctly clipped.
             */
            slice map_limits(size_t const length) const {
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
    };

    /*
     * Apply unary function to every element of input buffer, then place result in output buffer.
     *
     * Parameters
     * ----------
     * ufunc : F
     *     Unary function to apply.
     *     Must have signature "T2 f(T1 const&)".
     * in : ndarray<T1>* const
     *     N-D input buffer.
     * out : ndarray<T2>* const
     *     N-D output buffer.
     *     Must have same dimensions as input buffer.
     */
    template <typename F, typename T1, typename T2>
    void apply(F ufunc, ndarray<T1>* const in, ndarray<T2>* const out) {
        NDARRAY_ASSERT(in != nullptr, "Parameter[in] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(out != nullptr, "Parameter[out] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(in->shape() == out->shape(),
                       "Parameter[out] must have same dimensions as Parameter[in].");

        auto it_out = out->begin();
        for(auto it_in = in->begin(); it_in != in->end(); ++it_in, ++it_out) {
            *it_out = ufunc(*it_in);
        }
    }

    /*
     * Apply binary function to every element of input buffers, then place result in output buffer.
     *
     * Parameters
     * ----------
     * ufunc : F
     *     Binary function to apply.
     *     Must have signature "T2 f(T1 const&, T1 const&)".
     * in_1 : ndarray<T1>* const
     *     N-D input buffer (first argument).
     * in_2 : ndarray<T1>* const
     *     N-D input buffer (second argument).
     * out : ndarray<T2>* const
     *     N-D output buffer.
     *     Must have same dimensions as input buffers.
     *
     * Notes
     * -----
     * Input buffers are broadcasted together.
     */
    template <typename F, typename T1, typename T2>
    void apply(F ufunc, ndarray<T1>* const in_1, ndarray<T1>* const in_2, ndarray<T2>* const out) {
        NDARRAY_ASSERT(in_1 != nullptr, "Parameter[in_1] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(in_2 != nullptr, "Parameter[in_2] must point to a valid ndarray instance.");
        NDARRAY_ASSERT(out != nullptr, "Parameter[out] must point to a valid ndarray instance.");

        shape_t const correct_out_shape = predict_shape_broadcast(in_1->shape(), in_2->shape());
        NDARRAY_ASSERT(out->shape() == correct_out_shape,
                       "Parameter[in_1, in_2] do not broadcast to Parameter[out] dimensions.");

        ndarray<T1> in_1_bcast = in_1->broadcast_to(correct_out_shape);
        ndarray<T1> in_2_bcast = in_2->broadcast_to(correct_out_shape);
        auto it_out = out->begin();
        for(auto it_in_1 = in_1_bcast.begin(),
                 it_in_2 = in_2_bcast.begin();
            it_out != out->end();
            ++it_in_1, ++it_in_2, ++it_out) {
            *it_out = ufunc(*it_in_1, *it_in_2);
        }
    }

    /*
     * Apply reduction function along input buffer, then place result in output buffer.
     *
     * Parameters
     * ----------
     * ufunc : F
     *     Reduction function to apply.
     *     Must have signature "T2 f(T1 const&, T1 const&)" and assumed commutative/associative.
     * in : ndarray<T>* const
     *     (in_1, in_2, in_3) contiguous input buffer.
     * out : ndarray<T>* const
     *     (out_1, out_2, out_3) contiguous output buffer.
     *     Must have (out_[k] == in_[k]), except along `axis` where (out_[axis] == 1).
     * axis : size_t const
     *     Dimension along which to apply the reduction.
     * init : T const
     *     Initial reduction output.
     */
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

        std::vector<size_t> idx_out(3);  /* How to index the output. */
        std::vector<slice> selection(3); /* How to subsample the input. */
        std::vector<size_t> fix_axes;    /* Axes that are not reduced. */
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
                        selection[0] = slice();
                        selection[1] = slice(i, i + 1);
                        selection[2] = slice(j, j + 1);
                        idx_out[0] = 0;
                        idx_out[1] = i;
                        idx_out[2] = j;
                        break;

                    case 1:
                        selection[0] = slice(i, i + 1);
                        selection[1] = slice();
                        selection[2] = slice(j, j + 1);
                        idx_out[0] = i;
                        idx_out[1] = 0;
                        idx_out[2] = j;
                        break;

                    case 2:
                        selection[0] = slice(i, i + 1);
                        selection[1] = slice(j, j + 1);
                        selection[2] = slice();
                        idx_out[0] = i;
                        idx_out[1] = j;
                        idx_out[2] = 0;
                        break;
                }

                ndarray<T> sub_in = in->operator()(selection);
                out->operator[](idx_out) =
                    std::accumulate(sub_in.begin(), sub_in.end(), init, ufunc);
            }
        }
    }

    /*
     * Apply reduction function along input buffer, then place result in output buffer.
     *
     * Parameters
     * ----------
     * ufunc : F
     *     Reduction function to apply.
     *     Must have signature "T2 f(T1 const&, T1 const&)" and assumed commutative/associative.
     * in : ndarray<T>* const
     *     N-D input buffer.
     * out : ndarray<T>* const
     *     N-D contiguous output buffer.
     *     Must have (out->shape()[k] == in->shape()[k]), except along `axis` where (out->shape()[axis] == 1).
     * axis : size_t const
     *     Dimension along which to apply the reduction.
     * init : T const
     *     Initial reduction output.
     */
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
        shape_t shape_in3D(3);
        shape_t shape_out3D(3);
        if(axis == 0) {
            // in(a, b, c) -> in3D(a, 1, b*c)
            // out(1, b, c) -> out3D(1, 1, b*c)
            axis3D = 0;
            shape_in3D[0] = in->shape()[0];
            shape_in3D[1] = 1;
            shape_in3D[2] = std::accumulate(in->shape().begin() + 1, in->shape().end(),
                                            1, std::multiplies<size_t>());
            shape_out3D[0] = 1;
            shape_out3D[1] = 1;
            shape_out3D[2] = shape_in3D[2];
        } else if(axis == in->ndim() - 1) {
            // in(a, b, c) -> in3D(a*b, 1, c)
            // out(a, b, 1) -> out3D(a*b, 1, 1)
            axis3D = 2;
            shape_in3D[0] = std::accumulate(in->shape().begin(), in->shape().end() - 1,
                                            1, std::multiplies<size_t>());
            shape_in3D[1] = 1;
            shape_in3D[2] = in->shape()[in->ndim() - 1];
            shape_out3D[0] = shape_in3D[0];
            shape_out3D[1] = 1;
            shape_out3D[2] = 1;
        } else {
            // in(a, b, c, d, e) -> in3D(a*b, c, d*e)
            // out(a, b, 1, d, e) -> out3D(a*b, 1, d*e)
            axis3D = 1;
            shape_in3D[0] = std::accumulate(in->shape().begin(), in->shape().begin() + axis,
                                            1, std::multiplies<size_t>());;
            shape_in3D[1] = in->shape()[axis];
            shape_in3D[2] = std::accumulate(in->shape().end() - (in->ndim() - 1 - axis), in->shape().end(),
                                            1, std::multiplies<size_t>());;
            shape_out3D[0] = shape_in3D[0];
            shape_out3D[1] = 1;
            shape_out3D[2] = shape_in3D[2];
        }

        ndarray<T> in3D  = in->reshape(shape_in3D);
        ndarray<T> out3D = out->reshape(shape_out3D);
        reduce3D(ufunc, &in3D, &out3D, axis3D, init);
        // At this point `out` contains the result.
    }
}

namespace nd::util::interop {
    /*
     * Check if array can be mapped to an Eigen structure.
     *
     * Parameters
     * ----------
     * x : ndarray<T>* const
     *
     * Returns
     * -------
     * is_mappable : bool
     *
     * Notes
     * -----
     * `x` is mappable if the conditions below hold:
     *
     *     * `x` is 1d or 2d;
     *     * No two elements in `x` could overlap in any way.
     *       Overlaps may arise when using striding tricks (i.e. sliding windows, ...)
     *     * No negative strides(). (Eigen Bug 747. Should be fixed in Eigen v3.4)
     */
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

    /*
     * Eigen view on the data of an ndarray.
     *
     * Parameters
     * ----------
     * x : ndarray<T>* const
     *    (N,) or (N, M) array.
     * check_mappability : bool const
     *     If `true`, an exception is raised if mapping is impossible.
     * EigenFormat (template type)
     *     Must be one of {nd::mapA_t<T>, nd::mapM_t<T>}.
     *
     * Returns
     * -------
     * map : std::unique_ptr<EigenFormat>
     *     Eigen view on the array.
     *     (A pointer is returned to avoid instantiation of implicit EigenFormat copy-constructors.)
     *
     * Notes
     * -----
     * * The shape of the output depends on the shape of the input::
     *
     *     +==================+
     *     +    x    |  map   +
     *     +------------------+
     *     + (N,)    | (1, N) +
     *     + (N, M)  | (N, M) +
     *     +==================+
     *
     * * `check_mappability` may be set to `false` for performance reasons if certain a correct
     *   mapping is achievable. (i.e. if nd::util::interop::is_eigen_mappable() was called
     *   manually beforehand.)
     */
    template <typename T, typename EigenFormat>
    std::unique_ptr<EigenFormat> aseigenarray(ndarray<T>* const x,
                                              bool const check_mappability = true) {
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

    /*
     * Map Eigen result into an ndarray.
     *
     * Parameters
     * ----------
     * x : Eigen::DenseBase<Derived> const&
     *    (N, M) Eigen expression to evaluate.
     *
     * Returns
     * -------
     * y : ndarray<Derived::Scalar>
     *     (N, M) Ndarray containing the evaluation of `x`.
     *     `y` owns its own memory and can be used without restriction.
     */
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

namespace {
    /*
     * Send a (numeric) vector to output stream.
     */
    template <typename T>
    std::ostream& operator<<(std::ostream& os, std::vector<T> const& v) {
        os << "{";
        for(size_t i = 0; i < v.size() - 1; ++i) {
            os << v[i] << ", ";
        }
        os << v[v.size() - 1] << "}";

        return os;
    }

    /*
     * Send ndarray to output stream.
     *
     * Only 1d/2d arrays can be printed.
     */
    template <typename T>
    std::ostream& operator<<(std::ostream& os, nd::ndarray<T> const& x) {
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

#endif // _NDUTIL_HPP
