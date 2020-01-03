// ############################################################################
// _nditer.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDITER_HPP
#define _NDITER_HPP

#include <iterator>
#include <numeric>

#include "_ndtype.hpp"

namespace nd::util {
    template <typename T> void NDARRAY_ASSERT(bool const cond, T const& msg);
}

namespace nd {
    template <typename T> class ndarray;

    /*
     * LegacyForwardIterator to traverse ndarrays.
     */
    template <typename T>
    class ndarray_iterator {
        public:
            typedef std::forward_iterator_tag iterator_category;
            typedef T                                value_type;
            typedef std::ptrdiff_t              difference_type;
            typedef T*                                  pointer;
            typedef T&                                reference;

        private:
            ndarray<T>* m_iterable = nullptr;
            index_t     m_index;
            int         m_offset = 0;

            /*
             * Move iterator forward.
             *
             * Example
             * -------
             * Let `x` be an ndarray_iterator<T> on a (5, 3, 4) array,
             * with `m_index = {2, 0, 1}`::
             *
             *     x.advance();   // m_index == {2, 0, 2}
             *     x.advance();   // m_index == {2, 0, 3}
             *     x.advance();   // m_index == {2, 1, 0}
             *     x.advance();   // m_index == {2, 1, 1}
             */
            void advance() {
                size_t   const  ndim   = m_index.size();
                shape_t  const& shape  = m_iterable->shape();
                stride_t const& stride = m_iterable->strides();

                size_t carry = 1;
                for(size_t i = 0; (i < ndim) && (carry != 0); ++i) {
                    size_t const idx = (ndim - 1) - i;
                    size_t& cell = m_index[idx];

                    cell += carry;
                    m_offset += stride[idx];
                    if(cell == shape[idx]) {
                        // carry propagation
                        cell = 0;
                        carry = 1;
                        m_offset -= shape[idx] * stride[idx];
                    } else {
                        carry = 0;
                    }
                }

                if(carry == 1) {
                    // Leading dimension overflowed, therefore we encode out-of-boundness.
                    m_index = shape;
                    m_offset = std::inner_product(m_index.begin(), m_index.end(),
                                                  stride.begin(), int(0));
                }
            }

        public:
            ndarray_iterator() = delete;

            ndarray_iterator(ndarray<T>* const x):
                m_iterable(x),
                m_index(x->ndim(), 0),
                m_offset(0) {}

            ndarray_iterator(ndarray<T>* const x, index_t const& index):
                m_iterable(x),
                m_index(index) {
                    util::NDARRAY_ASSERT(index.size() == x->ndim(),
                                         "Index inconsistent with array rank.");
                    m_offset = std::inner_product(index.begin(), index.end(),
                                                  (*x).strides().begin(), int(0));
                }

            ndarray_iterator(ndarray_iterator<T> const& other):
                m_iterable(other.m_iterable),
                m_index(other.m_index),
                m_offset(other.m_offset) {}

            ~ndarray_iterator() {}

            ndarray_iterator<T>& operator=(ndarray_iterator<T> const& other) {
                m_iterable = other.m_iterable;
                m_index = other.m_index;
                m_offset = other.m_offset;

                return *this;
            }

            bool operator==(ndarray_iterator<T> const& other) const {
                bool const same_iterable = m_iterable->equals(*other.m_iterable);
                bool const same_index    = (m_index == other.m_index);
                bool const same_offset   = (m_offset == other.m_offset);

                return same_iterable && same_index && same_offset;
            }

            bool operator!=(ndarray_iterator<T> const& other) const {
                return !operator==(other);
            }

            T& operator*() const {
                byte_t* const head    = reinterpret_cast<byte_t*>(m_iterable->data());
                T*      const current = reinterpret_cast<T*>(head + m_offset);

                return *current;
            }

            ndarray_iterator<T>& operator++() {
                advance();
                return *this;
            }
    };
}

#endif // _NDITER_HPP
