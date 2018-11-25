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
#include <iterator>
#include <memory>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "_ndcontainer.hpp"
#include "_ndfunc.hpp"
#include "_nditer.hpp"
#include "_ndtype.hpp"
#include "_ndutil.hpp"

namespace nd {
    /*
     * Multi-dimensional array with NumPy semantics.
     */
    template <typename T>
    class ndarray {
        static_assert(byte_alignment >= sizeof(T),
                      "nd::byte_alignment too small to represent Type[T].");

        static constexpr bool is_bool = std::is_same<T, bool>::value;
        static constexpr bool is_int = std::is_integral<T>::value && (!is_bool);
        static constexpr bool is_float = std::is_floating_point<T>::value;
        static constexpr bool is_complex = (std::is_same<T, std::complex<float>>::value ||
                                            std::is_same<T, std::complex<double>>::value ||
                                            std::is_same<T, std::complex<long double>>::value);
        static constexpr bool is_arithmetic = is_bool || is_int || is_float || is_complex;
        static_assert(is_arithmetic, "Only arithmetic types are supported.");

        private:
            std::shared_ptr<ndcontainer> m_base = nullptr;
            byte_t*                      m_data = nullptr;
            shape_t                      m_shape;
            stride_t                     m_strides;  // byte offsets
            bool                         m_contiguous = true;

            bool verify_contiguous() const {
                stride_t contiguous_strides(m_shape.size(), 1);
                std::partial_sum(m_shape.rbegin(), m_shape.rend() - 1,
                                 contiguous_strides.rbegin() + 1, std::multiplies<int>());
                for(auto& s : contiguous_strides) {
                    s *= sizeof(T);
                }

                return m_strides == contiguous_strides;
            }

        public:
            /* Constructor ================================================= */
            ndarray() = delete;

            /*
             * Create (1,) array initialized with `scalar`.
             */
            ndarray(T const& scalar):
                m_base(std::make_shared<ndcontainer>(sizeof(T))),
                m_data(m_base->data()),
                m_shape({1}),
                m_strides({sizeof(T)}),
                m_contiguous(true) {
                    reinterpret_cast<T*>(m_data)[0] = scalar;
                }

            /*
             * Create array of dimensions `shape`.
             */
            ndarray(shape_t const& shape):
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

            ndarray(std::shared_ptr<ndcontainer> const& base,
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

            /*
             * Create shallow copy of `other`.
             *
             * Both :cpp:ptr:`this` and `other` point to the same memory buffer.
             */
            ndarray(ndarray<T> const& other):
                m_base(other.m_base),
                m_data(other.m_data),
                m_shape(other.m_shape),
                m_strides(other.m_strides),
                m_contiguous(other.m_contiguous) {}

            /*
             * Interpret pre-existing continuous memory block as an array.
             *
             * Parameters
             * ----------
             * data : byte_t* const
             *     Pointer to first element in array.
             * shape : nd::shape_t const&
             *     Number of elements per dimension.
             *
             * Notes
             * -----
             * The array does not take ownership of the memory buffer.
             * It is the responsibility of the user to free this resource.
             */
            ndarray(byte_t* const data, shape_t const& shape):
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

            /*
             * Interpret pre-existing strided memory block as an array.
             *
             * Parameters
             * ----------
             * data : byte_t* const
             *     Pointer to first element in array.
             * shape : nd::shape_t const&
             *     Number of elements per dimension.
             * strides : nd::stride_t const&
             *     Number of bytes to skip per dimension to reach the
             *     next element.
             *
             * Notes
             * -----
             * The array does not take ownership of the memory buffer.
             * It is the responsibility of the user to free this resource.
             */
            ndarray(byte_t* const data, shape_t const& shape, stride_t const& strides):
                m_base(nullptr),
                m_data(data),
                m_shape(shape),
                m_strides(strides) {
                    util::NDARRAY_ASSERT(shape.size() == strides.size(),
                                         "shape and strides must have the same length.");
                    m_contiguous = verify_contiguous();
                }

            ~ndarray() {}

            /* Property ==================================================== */
            std::shared_ptr<ndcontainer> const& base() const { return m_base; }

            /*
             * Returns
             * -------
             * data : T*
             *     Pointer to first element in array.
             */
            T* data() const { return reinterpret_cast<T*>(m_data); }

            /*
             * Returns
             * -------
             * shape : nd::shape_t const&
             *     Number of elements per dimension.
             */
            shape_t const& shape() const { return m_shape; }

            /*
             * Returns
             * -------
             * size : size_t
             *     Total number of elements in array.
             */
            size_t size() const {
                return std::accumulate(m_shape.begin(), m_shape.end(),
                                       1, std::multiplies<size_t>());
            }

            /*
             * Returns
             * -------
             * ndim : size_t
             *     Number of dimensions.
             */
            size_t ndim() const { return m_shape.size(); }

            /*
             * Returns
             * -------
             * strides : nd::stride_t const&
             *     Number of bytes to skip per dimension to reach
             *     the next element.
             */
            stride_t const& strides() const { return m_strides; }

            /*
             * Returns
             * -------
             * nbytes : size_t
             *     Number of bytes required to represent the data.
             *
             *     This is *not* the same thing as the number of bytes
             *     allocated in the buffer!
             *
             * Notes
             * -----
             * This method returns a byte-count based on the shape of the array.
             * The value returned will be wrong when using advanced striding tricks.
             */
            size_t nbytes() const {
                return std::accumulate(m_shape.begin(), m_shape.end(),
                                       sizeof(T), std::multiplies<size_t>());
            }

            /*
             * Returns
             * -------
             * same_memory : bool
             *     True if :cpp:ptr:`this` and `other` point to the same memory.
             */
            bool equals(ndarray<T> const& other) const {
                bool const same_data = (m_data == other.m_data);
                bool const same_shape = (m_shape == other.m_shape);
                bool const same_strides = (m_strides == other.m_strides);

                return same_data && same_shape && same_strides;
            }

            bool is_contiguous() const { return m_contiguous; }

            /* Index / Filter / Iterate ==================================== */
            /*
             * Select a specific entry in the array.
             *
             * Parameters
             * ----------
             * idx : std::vector<size_t> const&
             *     Index of element to extract.
             *
             * Returns
             * -------
             * elem : T&
             *     Extracted entry.
             */
            T& operator[](std::vector<size_t> const& idx) const {
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

            /*
             * Extract a sub-array.
             *
             * Parameters
             * ----------
             * spec : std::vector<nd::util::slice> const&
             *     Slice specification per dimension.
             *     If less entries than input-dimensions are given,
             *     then the trailing specifiers are set to `slice()`.
             *
             * Returns
             * -------
             * sub : ndarray<T>
             *     The sub-array.
             */
            ndarray<T> operator()(std::vector<util::slice> const& spec) const {
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

            /*
             * Returns
             * -------
             * iter : ndarray_iterator<T>
             *     Iterator at start of array.
             */
            ndarray_iterator<T> begin() const {
                ndarray<T>* const x = const_cast<ndarray<T>*>(this);
                return ndarray_iterator<T>(x);
            }

            /*
             * Returns
             * -------
             * iter : ndarray_iterator<T>
             *     Sentinel iterator marking end of array.
             */
            ndarray_iterator<T> end() const {
                ndarray<T>* const x = const_cast<ndarray<T>*>(this);
                return ndarray_iterator<T>(x, m_shape);
            }

            /*
             * Extract elements that satisfy condition.
             *
             * Parameters
             * ----------
             * mask : ndarray<bool> const&
             *     Boolean mask. Broadcasting rules apply.
             *
             * Returns
             * -------
             * out : ndarray<T>
             *     (N,) values where `mask` is :cpp:obj:`true`.
             */
            ndarray<T> where(ndarray<bool> const& mask) const {
                /*
                 * Suboptimal solution: use a vector to store the extracted elements,
                 * then perform a second copy into an ndarray<T> object.
                 */
                ndarray<bool> const mask_bcast = mask.broadcast_to(m_shape);
                auto it_this = this->begin();
                std::vector<T> buffer;
                for(auto it_mask = mask_bcast.begin();
                    it_mask != mask_bcast.end();
                    ++it_mask, ++it_this) {
                    if(*it_mask) {
                        buffer.push_back(*it_this);
                    }
                }

                ndarray<T> const out = r_(buffer);
                return out;
            }

            /*
             * Replace entries that satisfy condition.
             *
             * Parameters
             * ----------
             * mask : ndarray<bool> const&
             *     Boolean mask. Broadcasting rules apply.
             * x : ndarray<T> const&
             *     (N,) values to place where `mask` is :cpp:obj:`true`.
             */
            ndarray<T>& filter(ndarray<bool> const& mask, ndarray<T> const& x) const;

            /* Manipulation ================================================ */
            /*
             * Returns
             * -------
             * cpy : ndarray<T>
             *     Contiguous copy of :cpp:ptr:`this`.
             */
            ndarray<T> copy() const {
                ndarray<T> cpy(m_shape);

                for(ndarray_iterator<T> it_this = this->begin(),
                                        it_cpy = cpy.begin();
                    it_this != this->end();
                    ++it_this, ++it_cpy) {
                    *it_cpy = *it_this;
                }

                return cpy;
            }

            /*
             * Remove single-dimensional entries from the shape of an array.
             *
             * Returns
             * -------
             * squeezed : ndarray<T>
             *     The input array, but with all dimensions of length 1 removed.
             *     This is always :cpp:ptr:`this` itself or a view into it.
             */
            ndarray<T> squeeze() const {
                std::vector<size_t> idx;
                for(size_t i = 0; i < ndim(); ++i) {
                    size_t const& s = m_shape[i];
                    if(s == 1) {
                        idx.push_back(i);
                    }
                }

                return squeeze(idx);
            }

            /*
             * Remove single-dimensional entries from the shape of an array.
             *
             * Parameters
             * ----------
             * axes : std::vector<size_t> const&
             *     Subset of single-element dimensions to remove.
             *
             * Returns
             * -------
             * squeezed : ndarray<T>
             *     The input array, but with all dimensions of length 1 removed.
             *     This is always :cpp:ptr:`this` itself or a view into it.
             */
            ndarray<T> squeeze(std::vector<size_t> const& axes) const {
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

            /*
             * Returns an array containing the same data with a new shape.
             *
             * Returns
             * -------
             * reshaped : ndarray<T>
             *     This will be a new view object if possible; otherwise, it will be a copy.
             */
            ndarray<T> reshape(shape_t const& shape) const {
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

            /*
             * Return a contiguous flattened array.
             *
             * Returns
             * -------
             * rav : ndarray<T>
             *     1-D array containing the elements of :cpp:ptr:`this`.
             *     A copy is made only if needed.
             */
            ndarray<T> ravel() const {
                return ascontiguousarray(*this).reshape({size()});
            }

            /*
             * Broadcast to new shape.
             *
             * Returns
             * -------
             * bcast : ndarray<T>
             *     View on :cpp:ptr:`this` with the given shape.
             *     It is typically not contiguous.
             *     Furthermore, more than one element of a broadcasted array may
             *     refer to a single memory location.
             */
            ndarray<T> broadcast_to(shape_t const& shape) const {
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

            /*
             * Array view with axes transposed.
             *
             * Returns
             * -------
             * tr : ndarray<T>
             *     View on :cpp:ptr:`this` with axes reversed.
             */
            ndarray<T> transpose() const;

            /*
             * Array view with axes transposed.
             *
             * Parameters
             * ----------
             * axes : std::vector<size_t> const&
             *     Reordered dimension sequence.
             *
             * Returns
             * -------
             * tr : ndarray<T>
             *     View on :cpp:ptr:`this` with axes suitably permuted.
             */
            ndarray<T> transpose(std::vector<size_t> const& axes) const;

            /*
             * Element-wise static_cast<U>() of the array.
             *
             * Returns
             * -------
             * casted : ndarray<U>
             *     Copy of :cpp:ptr:`this` with element-wise static_cast<U>() of the input.
             */
            template <typename U>
            ndarray<U> cast() const;
            // template <typename T>
            // template <typename U>
            // ndarray<U> cast() const {
            //     ndarray<U> casted(this->shape());

            //     auto it_this = this->begin();
            //     auto it_cast = casted.begin();
            //     for(; it_cast != it_cast.end();
            //           ++it_cast, ++it_this) {
            //         *it_cast = static_cast<U>(*it_this);
            //     }

            //     return casted;
            // }

            /* Operator ==================================================== */
            /*
             * Copy RHS array contents into LHS array.
             * Broadcasting rules apply.
             */
            ndarray<T>& operator=(ndarray<T> const& other) {
                // Implemented here due to C++ restrictions.

                ndarray<T> const other_bcast = other.broadcast_to(m_shape);
                for(auto it = begin(), it_other = other_bcast.begin();
                    it != end();
                    ++it, ++it_other) {
                    *it = *it_other;
                }

                return *this;
            }

            ndarray<T>& operator+=(ndarray<T> const& other) {
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

            ndarray<T>& operator-=(ndarray<T> const& other) {
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

            ndarray<T>& operator*=(ndarray<T> const& other) {
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

            ndarray<T>& operator/=(ndarray<T> const& other) {
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

            ndarray<T>& operator%=(ndarray<T> const& other) {
                static_assert(is_int, "Only {int} types allowed.");

                ndarray<T> const other_bcast = other.broadcast_to(m_shape);
                for(auto it = begin(), it_other = other_bcast.begin();
                    it != end();
                    ++it, ++it_other) {
                    *it %= *it_other;
                }

                return *this;
            }

            ndarray<T>& operator&=(ndarray<T> const& other) {
                static_assert(is_int, "Only {int} types allowed.");

                ndarray<T> const other_bcast = other.broadcast_to(m_shape);
                for(auto it = begin(), it_other = other_bcast.begin();
                    it != end();
                    ++it, ++it_other) {
                    *it &= *it_other;
                }

                return *this;
            }

            ndarray<T>& operator|=(ndarray<T> const& other) {
                static_assert(is_int, "Only {int} types allowed.");

                ndarray<T> const other_bcast = other.broadcast_to(m_shape);
                for(auto it = begin(), it_other = other_bcast.begin();
                    it != end();
                    ++it, ++it_other) {
                    *it |= *it_other;
                }

                return *this;
            }

            ndarray<T>& operator^=(ndarray<T> const& other) {
                static_assert(is_int, "Only {int} types allowed.");

                ndarray<T> const other_bcast = other.broadcast_to(m_shape);
                for(auto it = begin(), it_other = other_bcast.begin();
                    it != end();
                    ++it, ++it_other) {
                    *it ^= *it_other;
                }

                return *this;
            }

            ndarray<T>& operator<<=(ndarray<T> const& other) {
                static_assert(is_int, "Only {int} types allowed.");

                ndarray<T> const other_bcast = other.broadcast_to(m_shape);
                for(auto it = begin(), it_other = other_bcast.begin();
                    it != end();
                    ++it, ++it_other) {
                    *it <<= *it_other;
                }

                return *this;
            }

            ndarray<T>& operator>>=(ndarray<T> const& other) {
                static_assert(is_int, "Only {int} types allowed.");

                ndarray<T> const other_bcast = other.broadcast_to(m_shape);
                for(auto it = begin(), it_other = other_bcast.begin();
                    it != end();
                    ++it, ++it_other) {
                    *it >>= *it_other;
                }

                return *this;
            }

            ndarray<T>& operator++() {
                static_assert(is_int, "Only {int} types allowed.");

                for(auto it = begin(); it != end(); ++it) {
                    ++(*it);
                }

                return *this;
            }

            ndarray<T>& operator--() {
                static_assert(is_int, "Only {int} types allowed.");

                for(auto it = begin(); it != end(); ++it) {
                    --(*it);
                }

                return *this;
            }

            ndarray<T> operator-() const {
                static_assert(is_int || is_float || is_complex,
                              "Only {int, float, complex} types allowed.");

                ndarray<T> y = copy();
                T* it = y.data();
                for(size_t i = 0; i < y.size(); ++i, ++it) {
                    (*it) = -(*it);
                }

                return y;
            }

            ndarray<T> operator~() const {
                static_assert(is_int, "Only {int} types allowed.");

                ndarray<T> y = copy();
                T* it = y.data();
                for(size_t i = 0; i < y.size(); ++i, ++it) {
                    (*it) = ~(*it);
                }

                return y;
            }

            ndarray<bool> operator!() const {
                static_assert(is_bool, "Only {bool} type allowed.");

                ndarray<bool> y = copy();
                bool* it = y.data();
                for(size_t i = 0; i < y.size(); ++i, ++it) {
                    (*it) = !(*it);
                }

                return y;
            }

            ndarray<T> operator+(ndarray<T> const& other) const {
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

            ndarray<T> operator-(ndarray<T> const& other) const {
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

            ndarray<T> operator*(ndarray<T> const& other) const {
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

            ndarray<T> operator/(ndarray<T> const& other) const {
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

            ndarray<T> operator%(ndarray<T> const& other) const {
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

            ndarray<T> operator&(ndarray<T> const& other) const {
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

            ndarray<T> operator|(ndarray<T> const& other) const {
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

            ndarray<T> operator^(ndarray<T> const& other) const {
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

            ndarray<T> operator<<(ndarray<T> const& other) const {
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

            ndarray<T> operator>>(ndarray<T> const& other) const {
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

            ndarray<bool> operator&&(ndarray<bool> const& other) const {
                static_assert(is_bool, "Only {bool} type allowed.");

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

            ndarray<bool> operator||(ndarray<bool> const& other) const {
                static_assert(is_bool, "Only {bool} type allowed.");

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

            ndarray<bool> operator==(ndarray<T> const& other) const {
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

            ndarray<bool> operator!=(ndarray<T> const& other) const {
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

            ndarray<bool> operator<(ndarray<T> const& other) const {
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

            ndarray<bool> operator<=(ndarray<T> const& other) const {
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

            ndarray<bool> operator>(ndarray<T> const& other) const {
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

            ndarray<bool> operator>=(ndarray<T> const& other) const {
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

            /* Mathematical Functions ====================================== */
            /*
             * Array view on real component of complex-valued arrays.
             */
            ndarray<T> real() const;

            /*
             * Array view on imag component of complex-valued arrays.
             */
            ndarray<T> imag() const;

            /*
             * Element-wise conjugation of complex-valued arrays.
             */
            ndarray<T> conj() const;
    };
}

#endif // _NDARRAY_HPP