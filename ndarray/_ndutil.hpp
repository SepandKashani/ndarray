// ############################################################################
// _ndutil.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDUTIL_HPP
#define _NDUTIL_HPP

#include <limits>
#include <ostream>

#include "_ndtype.hpp"

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
    void NDARRAY_ASSERT(bool const cond, T const& msg);

    /*
     * Send shape information to output stream.
     */
    inline std::ostream& operator<<(std::ostream& os, shape_t const& shape);

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
    inline shape_t predict_shape(shape_t const& lhs, shape_t const& rhs);

    /*
     * NumPy-like slice object.
     */
    class slice {
        private:
            int m_start      = 0;
            int m_stop       = std::numeric_limits<int>::max();
            int m_step       = 1;

        public:
            slice();
            slice(int const stop);
            slice(int const start, int const stop);
            slice(int const start, int const stop, int const step);
            ~slice();

            int start() const;
            int stop() const;
            int step() const;

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
            slice map_limits(size_t const length) const;
    };
}}
#endif // _NDUTIL_HPP
