// ############################################################################
// _nditer.cpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDITER_CPP
#define _NDITER_CPP

#include <numeric>

#include "_ndarray.hpp"
#include "_nditer.hpp"
#include "_ndtype.hpp"
#include "_ndutil.hpp"

namespace nd {
    template <typename T>
    inline void ndarray_iterator<T>::advance() {
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

    template <typename T>
    inline ndarray_iterator<T>::ndarray_iterator(ndarray<T>* const x):
        m_iterable(x),
        m_index(x->ndim(), 0),
        m_offset(0) {}

    template <typename T>
    inline ndarray_iterator<T>::ndarray_iterator(ndarray<T>* const x, std::vector<size_t> index):
        m_iterable(x),
        m_index(index) {
            util::NDARRAY_ASSERT(index.size() == x->ndim(),
                                 "Index inconsistent with array rank.");
            m_offset = std::inner_product(index.begin(), index.end(),
                                         (x->strides()).begin(), int(0));
        }

    template <typename T>
    inline ndarray_iterator<T>::ndarray_iterator(ndarray_iterator<T> const& other):
        m_iterable(other.m_iterable),
        m_index(other.m_index),
        m_offset(other.m_offset) {}

    template <typename T>
    inline ndarray_iterator<T>::~ndarray_iterator() {}

    template <typename T>
    inline ndarray_iterator<T>& ndarray_iterator<T>::operator=(ndarray_iterator<T> const& other) {
        m_iterable = other.m_iterable;
        m_index = other.m_index;
        m_offset = other.m_offset;

        return *this;
    }

    template <typename T>
    inline bool ndarray_iterator<T>::operator==(ndarray_iterator<T> const& other) const {
        bool const same_iterable = m_iterable->equals(*(other.m_iterable));
        bool const same_index    = (m_index == other.m_index);
        bool const same_offset   = (m_offset == other.m_offset);

        return same_iterable && same_index && same_offset;
    }

    template <typename T>
    inline bool ndarray_iterator<T>::operator!=(ndarray_iterator<T> const& other) const {
        return !((*this) == other);
    }

    template <typename T>
    inline T& ndarray_iterator<T>::operator*() const {
        byte_t* const head    = reinterpret_cast<byte_t*>(m_iterable->data());
        T*      const current = reinterpret_cast<T*>(head + m_offset);

        return *current;
    }

    template <typename T>
    inline ndarray_iterator<T>& ndarray_iterator<T>::operator++() {
        advance();
        return *this;
    }
}

#endif // _NDITER_CPP
