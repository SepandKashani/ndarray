// ############################################################################
// _ndutil.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDUTIL_HPP
#define _NDUTIL_HPP

#include <algorithm>
#include <limits>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "_ndtype.hpp"
#include "_ndutil.hpp"

namespace nd { template <typename T> class ndarray; }

namespace nd { namespace util {
    /*
     * assert()-like statement that does not deactivate in Release mode.
     *
     * Parameters
     * ----------
     * cond : bool
     * msg : T const&
     */
    template <typename T>
    void NDARRAY_ASSERT(bool const cond, T const& msg) {
        if(!cond) {
            throw std::runtime_error(msg);
        }
    }

    /*
     * Send shape information to output stream.
     */
    inline std::ostream& operator<<(std::ostream& os, shape_t const& shape) {
        os << "{";
        for(size_t i = 0; i < shape.size() - 1; ++i) {
            os << shape[i] << ", ";
        }
        os << shape[shape.size() - 1] << "}";

        return os;
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
    inline shape_t predict_shape(shape_t const& lhs, shape_t const& rhs) {
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
     * NumPy-like slice object.
     */
    class slice {
        private:
            int m_start      = 0;
            int m_stop       = std::numeric_limits<int>::max();
            int m_step       = 1;

        public:
            slice() {};

            slice(int const stop):
                m_start(0),
                m_stop(stop),
                m_step(1) {}

            slice(int const start, int const stop):
                m_start(start),
                m_stop(stop),
                m_step(1) {}

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

        shape_t const correct_out_shape = predict_shape(in_1->shape(), in_2->shape());
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
}}

#endif // _NDUTIL_HPP
