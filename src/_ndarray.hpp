// ############################################################################
// _ndarray.hpp
// ============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDARRAY_HPP
#define _NDARRAY_HPP

#include <memory>
#include <type_traits>
#include <vector>

#include "_ndcontainer.hpp"
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

            bool verify_contiguous() const;

        public:
            /* Constructor ================================================= */
            ndarray() = delete;

            /*
             * Create (1,) array initialized with `scalar`.
             */
            ndarray(T const& scalar);

            /*
             * Create array of dimensions `shape`.
             */
            ndarray(shape_t const& shape);

            ndarray(std::shared_ptr<ndcontainer> const& base,
                    byte_t* const data,
                    shape_t const& shape,
                    stride_t const& strides);

            /*
             * Create shallow copy of `other`.
             *
             * Both :cpp:ptr:`this` and `other` point to the same memory buffer.
             */
            ndarray(ndarray<T> const& other);

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
            ndarray(byte_t* const data, shape_t const& shape);

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
            ndarray(byte_t* const data, shape_t const& shape, stride_t const& strides);

            ~ndarray();

            /* Property ==================================================== */
            std::shared_ptr<ndcontainer> const& base() const;

            /*
             * Returns
             * -------
             * data : T*
             *     Pointer to first element in array.
             */
            T* data() const;

            /*
             * Returns
             * -------
             * shape : nd::shape_t const&
             *     Number of elements per dimension.
             */
            shape_t const& shape() const;

            /*
             * Returns
             * -------
             * size : size_t
             *     Total number of elements in array.
             */
            size_t size() const;

            /*
             * Returns
             * -------
             * ndim : size_t
             *     Number of dimensions.
             */
            size_t ndim() const;

            /*
             * Returns
             * -------
             * strides : nd::stride_t const&
             *     Number of bytes to skip per dimension to reach
             *     the next element.
             */
            stride_t const& strides() const;

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
            size_t nbytes() const;

            /*
             * Returns
             * -------
             * same_memory : bool
             *     True if :cpp:ptr:`this` and `other` point to the same memory.
             */
            bool equals(ndarray<T> const& other) const;

            bool is_contiguous() const;

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
            T& operator[](std::vector<size_t> const& idx) const;

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
            ndarray<T> operator()(std::vector<util::slice> const& spec) const;

            /*
             * Returns
             * -------
             * iter : ndarray_iterator<T>
             *     Iterator at start of array.
             */
            ndarray_iterator<T> begin() const;

            /*
             * Returns
             * -------
             * iter : ndarray_iterator<T>
             *     Sentinel iterator marking end of array.
             */
            ndarray_iterator<T> end() const;

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
            ndarray<T> where(ndarray<bool> const& mask) const;

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
            ndarray<T> copy() const;

            /*
             * Remove single-dimensional entries from the shape of an array.
             *
             * Returns
             * -------
             * squeezed : ndarray<T>
             *     The input array, but with all dimensions of length 1 removed.
             *     This is always :cpp:ptr:`this` itself or a view into it.
             */
            ndarray<T> squeeze() const;

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
            ndarray<T> squeeze(std::vector<size_t> const& axes) const;

            /*
             * Returns an array containing the same data with a new shape.
             *
             * Returns
             * -------
             * reshaped : ndarray<T>
             *     This will be a new view object if possible; otherwise, it will be a copy.
             */
            ndarray<T> reshape(shape_t const& shape) const;

            /*
             * Return a contiguous flattened array.
             *
             * Returns
             * -------
             * rav : ndarray<T>
             *     1-D array containing the elements of :cpp:ptr:`this`.
             *     A copy is made only if needed.
             */
            ndarray<T> ravel() const;

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
            ndarray<T> broadcast_to(shape_t const& shape) const;

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

            ndarray<T>&   operator+= (ndarray<T> const& other);
            ndarray<T>&   operator-= (ndarray<T> const& other);
            ndarray<T>&   operator*= (ndarray<T> const& other);
            ndarray<T>&   operator/= (ndarray<T> const& other);
            ndarray<T>&   operator%= (ndarray<T> const& other);
            ndarray<T>&   operator&= (ndarray<T> const& other);
            ndarray<T>&   operator|= (ndarray<T> const& other);
            ndarray<T>&   operator^= (ndarray<T> const& other);
            ndarray<T>&   operator<<=(ndarray<T> const& other);
            ndarray<T>&   operator>>=(ndarray<T> const& other);
            ndarray<T>&   operator++ ();
            ndarray<T>&   operator-- ();
            ndarray<T>    operator-  () const;
            ndarray<T>    operator~  () const;
            ndarray<bool> operator!  () const;
            ndarray<T>    operator+  (ndarray<T> const& other) const;
            ndarray<T>    operator-  (ndarray<T> const& other) const;
            ndarray<T>    operator*  (ndarray<T> const& other) const;
            ndarray<T>    operator/  (ndarray<T> const& other) const;
            ndarray<T>    operator%  (ndarray<T> const& other) const;
            ndarray<T>    operator&  (ndarray<T> const& other) const;
            ndarray<T>    operator|  (ndarray<T> const& other) const;
            ndarray<T>    operator^  (ndarray<T> const& other) const;
            ndarray<T>    operator<< (ndarray<T> const& other) const;
            ndarray<T>    operator>> (ndarray<T> const& other) const;
            ndarray<bool> operator&& (ndarray<bool> const& other) const;
            ndarray<bool> operator|| (ndarray<bool> const& other) const;
            ndarray<bool> operator== (ndarray<T> const& other) const;
            ndarray<bool> operator!= (ndarray<T> const& other) const;
            ndarray<bool> operator<  (ndarray<T> const& other) const;
            ndarray<bool> operator<= (ndarray<T> const& other) const;
            ndarray<bool> operator>  (ndarray<T> const& other) const;
            ndarray<bool> operator>= (ndarray<T> const& other) const;

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
