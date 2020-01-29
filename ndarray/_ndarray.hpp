// ############################################################################
// _ndarray.hpp
// ============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDARRAY_HPP
#define _NDARRAY_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "_ndforward.hpp"

namespace nd {
    template <typename T>
    bool ndarray<T>::verify_contiguous() const {
        stride_t str_cont(m_shape.size(), 1);
        std::partial_sum(m_shape.rbegin(), m_shape.rend() - 1,
                         str_cont.rbegin() + 1, std::multiplies<int>());
        for(auto& s : str_cont) {
            s *= sizeof(T);
        }

        return m_strides == str_cont;
    }

    template <typename T>
    ndarray<T>::~ndarray() {}

    template <typename T>
    ndarray<T>::ndarray(shape_t const& shape):
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
    ndarray<T>::ndarray(std::shared_ptr<ndcontainer> const& base,
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
    ndarray<T>::ndarray(ndarray<T> const& other):
        m_base(other.m_base),
        m_data(other.m_data),
        m_shape(other.m_shape),
        m_strides(other.m_strides),
        m_contiguous(other.m_contiguous) {}

    template <typename T>
    ndarray<T>::ndarray(byte_t* const data, shape_t const& shape):
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
    ndarray<T>::ndarray(byte_t* const data,
                        shape_t const& shape,
                        stride_t const& strides):
        m_base(nullptr),
        m_data(data),
        m_shape(shape),
        m_strides(strides) {
        util::NDARRAY_ASSERT(shape.size() == strides.size(),
                             "shape and strides must have the same length.");
        m_contiguous = verify_contiguous();
    }

    template <typename T>
    std::shared_ptr<ndcontainer> const& ndarray<T>::base() const {
        return m_base;
    }

    template <typename T>
    T* ndarray<T>::data() const {
        return reinterpret_cast<T*>(m_data);
    }

    template <typename T>
    shape_t const& ndarray<T>::shape() const {
        return m_shape;
    }

    template <typename T>
    size_t ndarray<T>::size() const {
        return std::accumulate(m_shape.begin(), m_shape.end(),
                               1, std::multiplies<size_t>());
    }

    template <typename T>
    size_t ndarray<T>::ndim() const {
        return m_shape.size();
    }

    template <typename T>
    stride_t const& ndarray<T>::strides() const {
        return m_strides;
    }

    template <typename T>
    size_t ndarray<T>::nbytes() const {
        return std::accumulate(m_shape.begin(), m_shape.end(),
                               sizeof(T), std::multiplies<size_t>());
    }

    template <typename T>
    bool ndarray<T>::equals(ndarray<T> const& other) const {
        bool const same_data = (m_data == other.m_data);
        bool const same_shape = (m_shape == other.m_shape);
        bool const same_strides = (m_strides == other.m_strides);

        return same_data && same_shape && same_strides;
    }

    template <typename T>
    bool ndarray<T>::is_contiguous() const {
        return m_contiguous;
    }

    template <typename T>
    T& ndarray<T>::operator[](index_t const& idx) const {
        int const offset = std::inner_product(idx.begin(), idx.end(),
                                              m_strides.begin(), int(0));
        byte_t* const addr = m_data + offset;
        return reinterpret_cast<T*>(addr)[0];
    }

    template <typename T>
    T& ndarray<T>::at(index_t const& idx) const {
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
    ndarray<T> ndarray<T>::operator()(std::vector<util::slice> const& spec) const {
        stride_t stride = m_strides;
        for(size_t i = 0; i < spec.size(); ++i) {
            stride[i] *= spec[i].step();
        }

        shape_t shape = m_shape;
        for(size_t i = 0; i < spec.size(); ++i) {
            auto s = spec[i].map_limits(m_shape[i]);
            int const length = std::abs(s.stop() - s.start()) - 1;
            shape[i] = ((s.start() != s.stop()) *
                        (1 + std::abs(length / s.step())));
        }

        index_t idx(m_shape.size(), 0);
        for(size_t i = 0; i < spec.size(); ++i) {
            auto s = spec[i].map_limits(m_shape[i]);
            idx[i] = s.start();
        }
        int const offset = std::inner_product(idx.begin(), idx.end(),
                                              m_strides.begin(), int(0));
        byte_t* const data = m_data + offset;

        return ndarray<T>(m_base, data, shape, stride);
    }

    template <typename T>
    ndarray_iterator<T> ndarray<T>::begin() const {
        ndarray<T>* const x = const_cast<ndarray<T>*>(this);
        return ndarray_iterator<T>(x);
    }

    template <typename T>
    ndarray_iterator<T> ndarray<T>::end() const {
        ndarray<T>* const x = const_cast<ndarray<T>*>(this);
        return ndarray_iterator<T>(x, m_shape);
    }

    template <typename T>
    ndarray<T> ndarray<T>::where(ndarray<bool> const& mask) const {
        auto const mask_bcast = mask.broadcast_to(m_shape);
        auto it_this = begin();
        std::vector<T> buffer;
        for(auto it_mask = mask_bcast.begin();
            it_mask != mask_bcast.end();
            ++it_mask, ++it_this) {
            if(*it_mask) {
                buffer.push_back(*it_this);
            }
        }

        ndarray<T> out({buffer.size()});
        std::copy_n(buffer.begin(), buffer.size(), out.begin());
        return out;
    }

    template <typename T>
    ndarray<T>& ndarray<T>::filter(ndarray<bool> const& mask,
                                   ndarray<T> const& x) {
        util::NDARRAY_ASSERT(x.ndim() == 1, "Parameter[x] must be 1-D.");

        auto const mask_bcast = mask.broadcast_to(m_shape);
        auto it_mask = mask_bcast.begin();
        auto it_this = begin();

        bool const broadcast_mode = (x.size() == 1);
        if(broadcast_mode) {
            T const& xx = x.data()[0];
            for(;
                it_mask != mask_bcast.end();
                ++it_mask, ++it_this) {
                if(*it_mask) {
                    *it_this = xx;
                }
            }
        } else {
            size_t N_modified = 0;
            for(auto it_x = x.begin();
                it_mask != mask_bcast.end();
                ++it_mask, ++it_this) {
                if(*it_mask) {
                    *it_this = *it_x;
                    ++N_modified;
                    ++it_x;
                }
            }

            util::NDARRAY_ASSERT(N_modified == x.size(),
                                 "Parameter[mask] does not have N true-valued cells.");
        }

        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::filter(ndarray<bool> const& mask,
                                   T const& x) {
        ndarray<T> _x({1});
        _x.data()[0] = x;
        return filter(mask, _x);
    }

    template <typename T>
    ndarray<T> ndarray<T>::copy() const {
        ndarray<T> cpy(m_shape);
        std::copy(begin(), end(), cpy.data());
        return cpy;
    }

    template <typename T>
    ndarray<T> ndarray<T>::squeeze(std::vector<size_t> const& axes) const {
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

            shape_t sh_squeezed;
            stride_t str_squeezed;
            for(size_t const& axis : kept_axes) {
                sh_squeezed.push_back(m_shape[axis]);
                str_squeezed.push_back(m_strides[axis]);
            }

            ndarray<T> squeezed(m_base, m_data, sh_squeezed, str_squeezed);
            return squeezed;
        }
    }

    template <typename T>
    ndarray<T> ndarray<T>::squeeze() const {
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
    ndarray<T> ndarray<T>::reshape(shape_t const& shape) const {
        size_t const new_size = std::accumulate(shape.begin(), shape.end(),
                                                1, std::multiplies<size_t>());
        if(new_size != size()) {
            std::stringstream error_msg;
            error_msg << "Cannot reshape array of size " << size()
                      << " into shape " << shape << ".\n";
            util::NDARRAY_ASSERT(false, error_msg.str());
        }

        // Compute new strides
        stride_t new_strides(shape.size(), 1);
        std::partial_sum(shape.rbegin(), shape.rend() - 1,
                         new_strides.rbegin() + 1, std::multiplies<int>());
        for(auto& s : new_strides) {
            s *= sizeof(T);
        }

        auto reshaped = ascontiguousarray(*this);
        reshaped.m_shape = shape;
        reshaped.m_strides = new_strides;

        return reshaped;
    }

    template <typename T>
    ndarray<T> ndarray<T>::ravel() const {
        return ascontiguousarray(*this).reshape({size()});
    }

    template <typename T>
    ndarray<T> ndarray<T>::broadcast_to(shape_t const& shape) const {
        auto sh_out = util::predict_shape_broadcast(m_shape, shape);
        if(sh_out != shape) {
            std::stringstream error_msg;
            error_msg << "Cannot broadcast array of shape " << m_shape
                      << " to " << shape << ".\n";
            util::NDARRAY_ASSERT(false, error_msg.str());
        }

        if(sh_out == m_shape) {
            return *this;
        } else {
            stride_t str_bcast(shape.size(), 0);
            for(size_t i = 0; i < m_shape.size(); ++i) {
                size_t const new_idx = (shape.size() - 1) - i;
                size_t const old_idx = (m_shape.size() - 1) - i;

                if(m_shape[old_idx] > 1) {
                    str_bcast[new_idx] = m_strides[old_idx];
                }
            }

            ndarray<T> bcast(m_base, m_data, shape, str_bcast);
            return bcast;
        }
    }

    template <typename T>
    ndarray<T> ndarray<T>::transpose(std::vector<size_t> const& axes) const {
        std::set<size_t> const _axes(axes.begin(), axes.end());
        size_t const max_axes = *_axes.crbegin();
        util::NDARRAY_ASSERT((_axes.size() == ndim()) &&
                             (max_axes == (ndim() - 1)),
                             "Parameter[axes] don't match array.");

        shape_t sh;
        stride_t str;
        for(size_t const& ax : axes) {
            sh.push_back(m_shape[ax]);
            str.push_back(m_strides[ax]);
        }

        ndarray<T> tr(m_base, m_data, sh, str);
        return tr;
    }

    template <typename T>
    ndarray<T> ndarray<T>::transpose() const {
        std::vector<size_t> axes(ndim());
        std::iota(axes.rbegin(), axes.rend(), 0);

        return transpose(axes);
    }

    template <typename T>
    template <typename U>
    ndarray<U> ndarray<T>::cast() const {
        ndarray<U> casted(m_shape);

        auto ufunc = [](T const& x) -> U { return static_cast<U>(x); };
        std::transform(begin(), end(), casted.begin(), ufunc);

        return casted;
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator=(ndarray<T> const& other) {
        auto const other_bcast = other.broadcast_to(m_shape);
        std::copy_n(other_bcast.begin(), size(), begin());
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator+=(ndarray<T> const& other) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        util::apply(std::plus<T>(), this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator+=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator+=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator-=(ndarray<T> const& other) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        util::apply(std::minus<T>(), this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator-=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator-=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator*=(ndarray<T> const& other) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        util::apply(std::multiplies<T>(), this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator*=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator*=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator/=(ndarray<T> const& other) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        util::apply(std::divides<T>(), this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator/=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator/=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator%=(ndarray<T> const& other) {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        util::apply(std::modulus<T>(), this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator%=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator%=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator&=(ndarray<T> const& other) {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        util::apply(std::bit_and<T>(), this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator&=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator&=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator|=(ndarray<T> const& other) {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        util::apply(std::bit_or<T>(), this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator|=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator|=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator^=(ndarray<T> const& other) {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        util::apply(std::bit_xor<T>(), this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator^=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator^=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator<<=(ndarray<T> const& other) {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto ufunc = [](T const& _x, T const& _y) -> T { return _x << _y; };
        util::apply(ufunc, this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator<<=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator<<=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator>>=(ndarray<T> const& other) {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto ufunc = [](T const& _x, T const& _y) -> T { return _x >> _y; };
        util::apply(ufunc, this, const_cast<ndarray<T>*>(&other), this);
        return *this;
    }
    template <typename T>
    ndarray<T>& ndarray<T>::operator>>=(T const& other) {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator>>=(_other);
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator++() {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto ufunc = [](T const& _x) -> T { return _x + 1; };
        util::apply(ufunc, this, this);
        return *this;
    }

    template <typename T>
    ndarray<T>& ndarray<T>::operator--() {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto ufunc = [](T const& _x) -> T { return _x - 1; };
        util::apply(ufunc, this, this);
        return *this;
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator-() const {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        ndarray<T> y(m_shape);
        util::apply(std::negate<T>(), const_cast<ndarray<T>*>(this), &y);
        return y;
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator~() const {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        ndarray<T> y(m_shape);
        util::apply(std::bit_not<T>(), const_cast<ndarray<T>*>(this), &y);
        return y;
    }

    template <typename T>
    ndarray<bool> ndarray<T>::operator!() const {
        static_assert(is_bool<T>(), "Only {bool} type allowed.");

        ndarray<bool> y(m_shape);
        util::apply(std::logical_not<bool>(), const_cast<ndarray<bool>*>(this), &y);
        return y;
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator+(ndarray<T> const& other) const {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);
        util::apply(std::plus<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator+(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator+(_other);
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator-(ndarray<T> const& other) const {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);
        util::apply(std::minus<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator-(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator-(_other);
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator*(ndarray<T> const& other) const {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);
        util::apply(std::multiplies<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator*(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator*(_other);
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator/(ndarray<T> const& other) const {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);
        util::apply(std::divides<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator/(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator/(_other);
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator%(ndarray<T> const& other) const {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);
        util::apply(std::modulus<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator%(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator%(_other);
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator&(ndarray<T> const& other) const {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);
        util::apply(std::bit_and<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator&(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator&(_other);
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator|(ndarray<T> const& other) const {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);
        util::apply(std::bit_or<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator|(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator|(_other);
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator^(ndarray<T> const& other) const {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);
        util::apply(std::bit_xor<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator^(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator^(_other);
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator<<(ndarray<T> const& other) const {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);

        auto ufunc = [](T const& _x, T const& _y) -> T { return _x << _y; };
        util::apply(ufunc,
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator<<(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator<<(_other);
    }

    template <typename T>
    ndarray<T> ndarray<T>::operator>>(ndarray<T> const& other) const {
        static_assert(is_int<T>(), "Only {int} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<T> out(sh_out);

        auto ufunc = [](T const& _x, T const& _y) -> T { return _x >> _y; };
        util::apply(ufunc,
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<T> ndarray<T>::operator>>(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator>>(_other);
    }

    template <typename T>
    ndarray<bool> ndarray<T>::operator&&(ndarray<bool> const& other) const {
        static_assert(is_bool<T>(), "Only {bool} type allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<bool> out(sh_out);
        util::apply(std::logical_and<bool>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<bool> ndarray<T>::operator&&(bool const& other) const {
        ndarray<bool> _other({1});
        _other.data()[0] = other;
        return operator&&(_other);
    }

    template <typename T>
    ndarray<bool> ndarray<T>::operator||(ndarray<bool> const& other) const {
        static_assert(is_bool<T>(), "Only {bool} type allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<bool> out(sh_out);
        util::apply(std::logical_or<bool>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<bool> ndarray<T>::operator||(bool const& other) const {
        ndarray<bool> _other({1});
        _other.data()[0] = other;
        return operator||(_other);
    }

    template <typename T>
    ndarray<bool> ndarray<T>::operator==(ndarray<T> const& other) const {
        static_assert(is_bool<T>() || is_int<T>(), "Only {bool, int} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<bool> out(sh_out);
        util::apply(std::equal_to<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<bool> ndarray<T>::operator==(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator==(_other);
    }

    template <typename T>
    ndarray<bool> ndarray<T>::operator!=(ndarray<T> const& other) const {
        static_assert(is_bool<T>() || is_int<T>(), "Only {bool, int} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<bool> out(sh_out);
        util::apply(std::not_equal_to<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<bool> ndarray<T>::operator!=(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator!=(_other);
    }

    template <typename T>
    ndarray<bool> ndarray<T>::operator<(ndarray<T> const& other) const {
        static_assert(is_int<T>() || is_float<T>(),
                      "Only {int, float} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<bool> out(sh_out);
        util::apply(std::less<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<bool> ndarray<T>::operator<(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator<(_other);
    }

    template <typename T>
    ndarray<bool> ndarray<T>::operator<=(ndarray<T> const& other) const {
        static_assert(is_int<T>() || is_float<T>(),
                      "Only {int, float} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<bool> out(sh_out);
        util::apply(std::less_equal<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<bool> ndarray<T>::operator<=(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator<=(_other);
    }

    template <typename T>
    ndarray<bool> ndarray<T>::operator>(ndarray<T> const& other) const {
        static_assert(is_int<T>() || is_float<T>(),
                      "Only {int, float} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<bool> out(sh_out);
        util::apply(std::greater<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<bool> ndarray<T>::operator>(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator>(_other);
    }

    template <typename T>
    ndarray<bool> ndarray<T>::operator>=(ndarray<T> const& other) const {
        static_assert(is_int<T>() || is_float<T>(),
                      "Only {int, float} types allowed.");

        auto const& sh_out = util::predict_shape_broadcast(m_shape, other.m_shape);
        ndarray<bool> out(sh_out);
        util::apply(std::greater_equal<T>(),
                    const_cast<ndarray<T>*>(this),
                    const_cast<ndarray<T>*>(&other),
                    &out);

        return out;
    }
    template <typename T>
    ndarray<bool> ndarray<T>::operator>=(T const& other) const {
        ndarray<T> _other({1});
        _other.data()[0] = other;
        return operator>=(_other);
        }
}

#endif // _NDARRAY_HPP
