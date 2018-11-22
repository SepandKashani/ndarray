// ############################################################################
// _ndarray.cpp
// ============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDARRAY_CPP
#define _NDARRAY_CPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "_ndarray.hpp"
#include "_ndcontainer.hpp"
#include "_ndfunc.hpp"
#include "_ndtype.hpp"
#include "_ndutil.hpp"

namespace nd {
    template <typename T>
    inline bool ndarray<T>::verify_contiguous() const {
        stride_t contiguous_strides(m_shape.size(), 1);
        std::partial_sum(m_shape.rbegin(), m_shape.rend() - 1,
                         contiguous_strides.rbegin() + 1, std::multiplies<int>());
        for(auto& s : contiguous_strides) {
            s *= sizeof(T);
        }

        return m_strides == contiguous_strides;
    }

    template <typename T>
    inline ndarray<T>::ndarray(T const& scalar):
        m_base(std::make_shared<ndcontainer>(sizeof(T))),
        m_data(m_base->data()),
        m_shape({1}),
        m_strides({sizeof(T)}),
        m_contiguous(true) {
            reinterpret_cast<T*>(m_data)[0] = scalar;
        }

    template <typename T>
    inline ndarray<T>::ndarray(shape_t const& shape):
        m_base(std::make_shared<ndcontainer>(
               std::accumulate(shape.begin(), shape.end(),
                               sizeof(T), std::multiplies<size_t>()))),
        m_data(m_base->data()),
        m_shape(shape),
        m_strides(shape.size(), 1),
        m_contiguous(true) {
            std::partial_sum(m_shape.rbegin(), m_shape.rend() - 1,
                             m_strides.rbegin() + 1, std::multiplies<int>());
            for(auto& s : m_strides) {
                s *= sizeof(T);
            }
        }

    template <typename T>
    inline ndarray<T>::ndarray(std::shared_ptr<ndcontainer> const& base,
                               byte_t* const data,
                               shape_t const& shape,
                               stride_t const& strides):
        m_base(base),
        m_data(data),
        m_shape(shape),
        m_strides(strides) {
            util::NDARRAY_ASSERT(shape.size() == strides.size(),
                                 "shape and strides must have the same length.");
            m_contiguous = verify_contiguous();
        }

    template <typename T>
    inline ndarray<T>::ndarray(ndarray<T> const& other):
        m_base(other.m_base),
        m_data(other.m_data),
        m_shape(other.m_shape),
        m_strides(other.m_strides),
        m_contiguous(other.m_contiguous) {}

    template <typename T>
    inline ndarray<T>::ndarray(byte_t* const data, shape_t const& shape):
        m_base(nullptr),
        m_data(data),
        m_shape(shape),
        m_strides(shape.size(), 1),
        m_contiguous(true) {
            std::partial_sum(m_shape.rbegin(), m_shape.rend() - 1,
                             m_strides.rbegin() + 1, std::multiplies<int>());
            for(auto& s : m_strides) {
                s *= sizeof(T);
            }
        }

    template <typename T>
    inline ndarray<T>::ndarray(byte_t* const data, shape_t const& shape, stride_t const& strides):
        m_base(nullptr),
        m_data(data),
        m_shape(shape),
        m_strides(strides) {
            util::NDARRAY_ASSERT(shape.size() == strides.size(),
                                 "shape and strides must have the same length.");
            m_contiguous = verify_contiguous();
        }

    template <typename T>
    inline ndarray<T>::~ndarray() {}

    template <typename T>
    inline std::shared_ptr<ndcontainer> const& ndarray<T>::base() const {
        return m_base;
    }

    template <typename T>
    inline T* ndarray<T>::data() const {
        return reinterpret_cast<T*>(m_data);
    }

    template <typename T>
    inline shape_t const& ndarray<T>::shape() const {
        return m_shape;
    }

    template <typename T>
    inline size_t ndarray<T>::size() const {
        return std::accumulate(m_shape.begin(), m_shape.end(),
                               1, std::multiplies<size_t>());
    }

    template <typename T>
    inline size_t ndarray<T>::ndim() const {
        return m_shape.size();
    }

    template <typename T>
    inline stride_t const& ndarray<T>::strides() const {
        return m_strides;
    }

    template <typename T>
    inline size_t ndarray<T>::nbytes() const {
        return std::accumulate(m_shape.begin(), m_shape.end(),
                               sizeof(T), std::multiplies<size_t>());
    }

    template <typename T>
    inline bool ndarray<T>::equals(ndarray<T> const& other) const {
        bool const same_data = (m_data == other.m_data);
        bool const same_shape = (m_shape == other.m_shape);
        bool const same_strides = (m_strides == other.m_strides);

        return same_data && same_shape && same_strides;
    }

    template <typename T>
    inline bool ndarray<T>::is_contiguous() const {
        return m_contiguous;
    }

    template <typename T>
    inline T& ndarray<T>::operator[](std::vector<size_t> const& idx) const {
        util::NDARRAY_ASSERT(idx.size() == this->ndim(),
                             "Incomplete index: cannot select unique element.");

        for(size_t i = 0; i < idx.size(); ++i) {
            util::NDARRAY_ASSERT(idx[i] < m_shape[i],
                                 "Invalid index along dimension " +
                                 std::to_string(i) + ".");
        }

        int const offset = std::inner_product(idx.begin(), idx.end(),
                                              m_strides.begin(), int(0));
        byte_t* const addr = m_data + offset;
        return reinterpret_cast<T*>(addr)[0];
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator()(std::vector<util::slice> const& spec) const {
        stride_t stride = m_strides;
        for(size_t i = 0; i < spec.size(); ++i) {
            stride[i] *= spec[i].step();
        }

        shape_t shape = m_shape;
        for(size_t i = 0; i < spec.size(); ++i) {
            util::slice s = spec[i].map_limits(m_shape[i]);
            int const length = (std::abs(static_cast<int>(s.stop()) -
                                         static_cast<int>(s.start())) - 1);
            shape[i] = ((s.start() != s.stop()) *
                        (1 + std::abs(length / s.step())));
        }

        std::vector<int> idx(m_shape.size(), 0);
        for(size_t i = 0; i < spec.size(); ++i) {
            util::slice s = spec[i].map_limits(m_shape[i]);
            idx[i] = static_cast<int>(s.start());
        }
        int const offset = std::inner_product(idx.begin(), idx.end(),
                                              m_strides.begin(), int(0));
        byte_t* const data = m_data + offset;

        return ndarray<T>(m_base, data, shape, stride);
    }

    template <typename T>
    inline ndarray_iterator<T> ndarray<T>::begin() const {
        ndarray<T>* const x = const_cast<ndarray<T>*>(this);
        return ndarray_iterator<T>(x);
    }

    template <typename T>
    inline ndarray_iterator<T> ndarray<T>::end() const {
        ndarray<T>* const x = const_cast<ndarray<T>*>(this);
        return ndarray_iterator<T>(x, m_shape);
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::copy() const {
        ndarray<T> cpy(m_shape);

        for(ndarray_iterator<T> it_this = this->begin(),
                                it_cpy = cpy.begin();
            it_this != this->end();
            ++it_this, ++it_cpy) {
            *it_cpy = *it_this;
        }

        return cpy;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::squeeze() const {
        std::vector<size_t> idx;
        for(size_t i = 0; i < ndim(); ++i) {
            size_t const& s = m_shape[i];
            if(s == 1) {
                idx.push_back(i);
            }
        }

        return squeeze(idx);
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::squeeze(std::vector<size_t> const& axes) const {
        std::set<size_t> dropped_axes(axes.begin(), axes.end());
        for(size_t const& axis : dropped_axes) {
            util::NDARRAY_ASSERT(m_shape[axis] == 1,
                                 "Cannot select an axis to squeeze out which has size not equal to 1.");
        }

        size_t const n_dim = dropped_axes.size();
        if(n_dim == ndim()) {
            // Only a single element in the container.
            ndarray<T> squeezed(m_base, m_data, {1}, {sizeof(T)});
            return squeezed;
        } else if(n_dim == 0) {
            return *this;
        } else {
            std::vector<size_t> all_axes(ndim());
            std::iota(all_axes.begin(), all_axes.end(), size_t(0));
            std::set<size_t> kept_axes;
            std::set_difference(all_axes.begin(), all_axes.end(),
                                dropped_axes.begin(), dropped_axes.end(),
                                std::inserter(kept_axes, kept_axes.begin()));

            shape_t shape_squeezed;
            stride_t strides_squeezed;
            for(size_t const& axis : kept_axes) {
                shape_squeezed.push_back(m_shape[axis]);
                strides_squeezed.push_back(m_strides[axis]);
            }

            ndarray<T> squeezed(m_base, m_data, shape_squeezed, strides_squeezed);
            return squeezed;
        }
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::reshape(shape_t const& shape) const {
        size_t const new_size = std::accumulate(shape.begin(), shape.end(),
                                                1, std::multiplies<size_t>());
        std::stringstream error_msg;
        error_msg << "Cannot reshape array of size "
                  << std::to_string(new_size) << " into shape ";
        util::operator<<(error_msg, shape);
        error_msg << ".\n";
        util::NDARRAY_ASSERT(new_size == size(), error_msg.str());

        // Compute new strides
        stride_t new_strides(shape.size(), 1);
        std::partial_sum(shape.rbegin(), shape.rend() - 1,
                         new_strides.rbegin() + 1, std::multiplies<int>());
        for(auto& s : new_strides) {
            s *= sizeof(T);
        }

        ndarray<T> reshaped = ascontiguousarray(*this);
        reshaped.m_shape = shape;
        reshaped.m_strides = new_strides;

        return reshaped;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::ravel() const {
        return ascontiguousarray(*this).reshape({size()});
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::broadcast_to(shape_t const& shape) const {
        std::stringstream error_msg;
        error_msg << "Cannot broadcast array of shape ";
        util::operator<<(error_msg, m_shape);
        error_msg << " to ";
        util::operator<<(error_msg, shape);
        error_msg << ".\n";

        shape_t out_shape = util::predict_shape(m_shape, shape);
        util::NDARRAY_ASSERT(out_shape == shape, error_msg.str());

        if(out_shape == m_shape) {
            return *this;
        } else {
            stride_t bcast_stride(shape.size(), 0);
            for(size_t i = 0; i < m_shape.size(); ++i) {
                size_t const new_idx = (shape.size() - 1) - i;
                size_t const old_idx = (m_shape.size() - 1) - i;

                if(m_shape[old_idx] > 1) {
                    bcast_stride[new_idx] = m_strides[old_idx];
                }
            }

            ndarray<T> bcast(m_base, m_data, shape, bcast_stride);
            return bcast;
        }
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator+=(ndarray<T> const& other) {
        static_assert(is_int || is_float || is_complex,
                      "Only {int, float, complex} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it += *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator-=(ndarray<T> const& other) {
        static_assert(is_int || is_float || is_complex,
                      "Only {int, float, complex} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it -= *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator*=(ndarray<T> const& other) {
        static_assert(is_int || is_float || is_complex,
                      "Only {int, float, complex} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it *= *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator/=(ndarray<T> const& other) {
        static_assert(is_int || is_float || is_complex,
                      "Only {int, float, complex} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it /= *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator%=(ndarray<T> const& other) {
        static_assert(is_int, "Only {int} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it %= *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator&=(ndarray<T> const& other) {
        static_assert(is_int, "Only {int} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it &= *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator|=(ndarray<T> const& other) {
        static_assert(is_int, "Only {int} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it |= *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator^=(ndarray<T> const& other) {
        static_assert(is_int, "Only {int} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it ^= *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator<<=(ndarray<T> const& other) {
        static_assert(is_int, "Only {int} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it <<= *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator>>=(ndarray<T> const& other) {
        static_assert(is_int, "Only {int} types allowed.");

        ndarray<T> const other_bcast = other.broadcast_to(m_shape);
        for(auto it = begin(), it_other = other_bcast.begin();
            it != end();
            ++it, ++it_other) {
            *it >>= *it_other;
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator++() {
        static_assert(is_int, "Only {int} types allowed.");

        for(auto it = begin(); it != end(); ++it) {
            ++(*it);
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T>& ndarray<T>::operator--() {
        static_assert(is_int, "Only {int} types allowed.");

        for(auto it = begin(); it != end(); ++it) {
            --(*it);
        }

        return *this;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator-() const {
        static_assert(is_int || is_float || is_complex,
                      "Only {int, float, complex} types allowed.");

        ndarray<T> y = copy();
        T* it = y.data();
        for(size_t i = 0; i < y.size(); ++i, ++it) {
            (*it) = -(*it);
        }

        return y;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator~() const {
        static_assert(is_int, "Only {int} types allowed.");

        ndarray<T> y = copy();
        T* it = y.data();
        for(size_t i = 0; i < y.size(); ++i, ++it) {
            (*it) = ~(*it);
        }

        return y;
    }

    template <typename T>
    inline ndarray<bool> ndarray<T>::operator!() const {
        static_assert(is_bool, "Only {bool} type allowed.");

        ndarray<bool> y = copy();
        bool* it = y.data();
        for(size_t i = 0; i < y.size(); ++i, ++it) {
            (*it) = !(*it);
        }

        return y;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator+(ndarray<T> const& other) const {
        static_assert(is_int || is_float || is_complex,
                      "Only {int, float, complex} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) + (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator-(ndarray<T> const& other) const {
        static_assert(is_int || is_float || is_complex,
                      "Only {int, float, complex} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) - (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator*(ndarray<T> const& other) const {
        static_assert(is_int || is_float || is_complex,
                      "Only {int, float, complex} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) * (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator/(ndarray<T> const& other) const {
        static_assert(is_int || is_float || is_complex,
                      "Only {int, float, complex} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) / (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator%(ndarray<T> const& other) const {
        static_assert(is_int, "Only {int} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) % (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator&(ndarray<T> const& other) const {
        static_assert(is_int, "Only {int} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) & (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator|(ndarray<T> const& other) const {
        static_assert(is_int, "Only {int} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) | (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator^(ndarray<T> const& other) const {
        static_assert(is_int, "Only {int} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) ^ (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator<<(ndarray<T> const& other) const {
        static_assert(is_int, "Only {int} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) << (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<T> ndarray<T>::operator>>(ndarray<T> const& other) const {
        static_assert(is_int, "Only {int} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<T> out = ndarray<T>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) >> (*it_other);
        }

        return out;
    }

    template <>
    inline ndarray<bool> ndarray<bool>::operator&&(ndarray<bool> const& other) const {
        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<bool> const this_bcast = broadcast_to(shape_out);
        ndarray<bool> const other_bcast = other.broadcast_to(shape_out);

        ndarray<bool> out = ndarray<bool>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) && (*it_other);
        }

        return out;
    }


    template <>
    inline ndarray<bool> ndarray<bool>::operator||(ndarray<bool> const& other) const {
        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<bool> const this_bcast = broadcast_to(shape_out);
        ndarray<bool> const other_bcast = other.broadcast_to(shape_out);

        ndarray<bool> out = ndarray<bool>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) || (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<bool> ndarray<T>::operator==(ndarray<T> const& other) const {
        static_assert(is_bool || is_int, "Only {bool, int} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<bool> out = ndarray<bool>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) == (*it_other);
        }

        return out;
    }

    template <typename T>
    inline ndarray<bool> ndarray<T>::operator!=(ndarray<T> const& other) const {
        static_assert(is_bool || is_int, "Only {bool, int} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<bool> out = ndarray<bool>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) != (*it_other);
        }

        return out;
    }



    template <typename T>
    inline ndarray<bool> ndarray<T>::operator<(ndarray<T> const& other) const {
        static_assert(is_int || is_float,
                      "Only {int, float} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<bool> out = ndarray<bool>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) < (*it_other);
        }

        return out;
    }



    template <typename T>
    inline ndarray<bool> ndarray<T>::operator<=(ndarray<T> const& other) const {
        static_assert(is_int || is_float,
                      "Only {int, float} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<bool> out = ndarray<bool>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) <= (*it_other);
        }

        return out;
    }



    template <typename T>
    inline ndarray<bool> ndarray<T>::operator>(ndarray<T> const& other) const {
        static_assert(is_int || is_float,
                      "Only {int, float} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<bool> out = ndarray<bool>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) > (*it_other);
        }

        return out;
    }



    template <typename T>
    inline ndarray<bool> ndarray<T>::operator>=(ndarray<T> const& other) const {
        static_assert(is_int || is_float,
                      "Only {int, float} types allowed.");

        shape_t const& shape_out = util::predict_shape(m_shape, other.m_shape);
        ndarray<T> const this_bcast = broadcast_to(shape_out);
        ndarray<T> const other_bcast = other.broadcast_to(shape_out);

        ndarray<bool> out = ndarray<bool>(shape_out);
        for(auto it_this = this_bcast.begin(), it_other = other_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_this, ++it_other, ++it_out) {
            (*it_out) = (*it_this) >= (*it_other);
        }

        return out;
    }
}

#endif // _NDARRAY_CPP
