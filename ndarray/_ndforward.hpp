// ############################################################################
// _ndforward.hpp
// ==============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

/*
 * Top-level forward header with ::
 *
 *     * type aliases;
 *     * class public/private interfaces;
 *     * functions;
 */

#ifndef NDFORWARD_HPP
#define NDFORWARD_HPP

#include <complex>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <ostream>
#include <type_traits>
#include <vector>

#include "Eigen/Eigen"

namespace nd {  // Constants, low-level helpers, forward-declarations
    /*
     * Convenience types.
     */
    using byte_t = uint8_t;
    using cfloat = std::complex<float>;
    using cdouble = std::complex<double>;
    using cldouble = std::complex<long double>;

    /*
     * Memory buffer byte alignment.
     * Must be {2**k, k > 0} and equal sizeof(<largest type you want to use>).
     */
    constexpr size_t byte_alignment = sizeof(cldouble);

    /*
     * Ndarray information types.
     */
    using shape_t  = std::vector<size_t>;
    using index_t = std::vector<size_t>;
    using stride_t = std::vector<int>;

    /*
     * Static type checks helpers.
     */
    template <typename T> constexpr bool is_bool();
    template <typename T> constexpr bool is_int();
    template <typename T> constexpr bool is_signed_int();
    template <typename T> constexpr bool is_float();
    template <typename T> constexpr bool is_complex();
    template <typename T> constexpr bool is_arithmetic();

    /*
     * Interoperability types.
     *
     * These are used to interface with the Eigen3 library.
     */
    using mapS_t = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    template <typename T> using A_t = Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    template <typename T> using M_t = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    template <typename T> using mapA_t = Eigen::Map<A_t<T>, Eigen::Unaligned, mapS_t>;
    template <typename T> using mapM_t = Eigen::Map<M_t<T>, Eigen::Unaligned, mapS_t>;



    template <typename T> class ndarray;
    template <typename T> class ndarray_iterator;
}

namespace nd::util {  // Utility functions & classes
    /*
     * assert()-like statement that does not deactivate in Release mode.
     *
     * Parameters
     * ----------
     * cond : bool const
     * msg : T const&
     */
    template <typename T>
    void NDARRAY_ASSERT(bool const cond,
                        T const& msg);

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
    inline shape_t predict_shape_broadcast(shape_t const& lhs,
                                           shape_t const& rhs);

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
    inline shape_t predict_shape_reduction(shape_t const& shape,
                                           size_t const axis);

    /*
     * NumPy-like slice object.
     */
    class slice {
        private:
            int m_start      = 0;
            int m_stop       = std::numeric_limits<int>::max();
            int m_step       = 1;

        public:
            inline slice();
            inline ~slice();

            /*
             * Parameters
             * ----------
             * stop : int const
             *     Termination index (exclusive).
             */
            inline slice(int const stop);

            /*
             * Parameters
             * ----------
             * start : int const
             *     Initialisation index (inclusive).
             * stop : int const
             *     Termination index (exclusive).
             */
            inline slice(int const start,
                         int const stop);

            /*
             * Parameters
             * ----------
             * start : int const
             *     Initialisation index (inclusive).
             * stop : int const
             *     Termination index (exclusive). `-1` is a valid termination
             *     index to include the first element of an array.
             * step : int const
             *     Step size (non-zero).
             */
            inline slice(int const start,
                         int const stop,
                         int const step);

            inline int start() const;
            inline int stop() const;
            inline int step() const;

            /*
             * Set (ambiguous) slice limits to correct values for an array
             * dimension of length `length`.
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
            inline slice map_limits(size_t const length) const;
    };

    /*
     * Apply unary function to every element of input buffer, then place result
     * in output buffer.
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
    void apply(F ufunc,
               ndarray<T1>* const in,
               ndarray<T2>* const out);

    /*
     * Apply binary function to every element of input buffers, then place
     * result in output buffer.
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
    void apply(F ufunc,
               ndarray<T1>* const in_1,
               ndarray<T1>* const in_2,
               ndarray<T2>* const out);

    /*
     * Apply reduction function along input buffer, then place result in output
     * buffer.
     *
     * Parameters
     * ----------
     * ufunc : F
     *     Reduction function to apply.
     *     Must have signature "T2 f(T1 const&, T1 const&)" and assumed
     *     commutative/associative.
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
                  T const init);

    /*
     * Apply reduction function along input buffer, then place result in output
     * buffer.
     *
     * Parameters
     * ----------
     * ufunc : F
     *     Reduction function to apply.
     *     Must have signature "T2 f(T1 const&, T1 const&)" and assumed
     *     commutative/associative.
     * in : ndarray<T>* const
     *     N-D input buffer.
     * out : ndarray<T>* const
     *     N-D contiguous output buffer.
     *     Must have (out->shape()[k] == in->shape()[k]), except along `axis`
     *     where (out->shape()[axis] == 1).
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
                T const init);
}

namespace nd::util::interop {  // Eigen <-> ndarray interoperability functions
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
    bool is_eigen_mappable(ndarray<T>* const x);

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
     *     (A pointer is returned to avoid instantiation of implicit EigenFormat
     *     copy-constructors.)
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
     * * `check_mappability` may be set to `false` for performance reasons if
     *   certain a correct mapping is achievable. (i.e. if
     *   nd::util::interop::is_eigen_mappable() was called manually beforehand.)
     */
    template <typename T, typename EigenFormat>
    std::unique_ptr<EigenFormat> aseigenarray(ndarray<T>* const x,
                                              bool const check_mappability = true);

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
    ndarray<typename Derived::Scalar> asndarray(Eigen::ArrayBase<Derived> const& x);
}

namespace nd {  // Classes
    /*
     * Thin shell around an aligned memory buffer.
     */
    class ndcontainer {
        static_assert( (byte_alignment > 1) &&
                      ((byte_alignment & (byte_alignment - 1)) == 0),
                      "byte_alignment must be a power of 2.");

        private:
            byte_t* m_buffer     = nullptr;
            size_t  m_size       = 0;
            bool    m_own_memory = true;
            size_t  m_offset     = 0;

        public:
            inline ndcontainer() = delete;
            inline ndcontainer(ndcontainer const& other) = delete;
            inline ndcontainer(ndcontainer const&& other) = delete;
            inline ndcontainer(size_t const nbytes);
            inline ndcontainer(byte_t* const data,
                               size_t const nbytes);
            inline ~ndcontainer();

            inline byte_t* data() const;
            inline size_t nbytes() const;
            inline bool own_memory() const;
            inline void operator=(ndcontainer const& other) = delete;
    };



    /*
     * Multi-dimensional array with NumPy semantics.
     */
    template <typename T>
    class ndarray {
        static_assert(byte_alignment >= sizeof(T),
                      "nd::byte_alignment too small to represent Type[T].");
        static_assert(is_arithmetic<T>(), "Only arithmetic types are supported.");

        private:
            std::shared_ptr<ndcontainer> m_base = nullptr;
            byte_t*                      m_data = nullptr;
            shape_t                      m_shape;
            stride_t                     m_strides;  // byte offsets
            bool                         m_contiguous = true;

            bool verify_contiguous() const;

        public:
            ndarray() = delete;
            ~ndarray();

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
             * Interpret pre-existing contiguous memory block as an array.
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
            ndarray(byte_t* const data,
                    shape_t const& shape);

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
            ndarray(byte_t* const data,
                    shape_t const& shape,
                    stride_t const& strides);

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
             *     Number of bytes to skip per dimension to reach the next element.
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

            /*
             * Select a specific entry in the array.
             *
             * No bounds-checking is performed.
             *
             * Parameters
             * ----------
             * idx : nd::index_t const&
             *     Index of element to extract.
             *
             * Returns
             * -------
             * elem : T&
             *     Extracted entry.
             */
            T& operator[](index_t const& idx) const;

            /*
             * Select a specific entry in the array.
             *
             * Bounds-checking is performed.
             *
             * Parameters
             * ----------
             * idx : nd::index_t const&
             *     Index of element to extract.
             *
             * Returns
             * -------
             * elem : T&
             *     Extracted entry.
             */
            T& at(index_t const& idx) const;

            /*
             * Extract a sub-array.
             *
             * Parameters
             * ----------
             * spec : std::vector<nd::util::slice> const&
             *     Slice specification per dimension.
             *     If less entries than input-dimensions are given, then the
             *     trailing specifiers are set to `slice()`.
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
             *     Boolean mask containing `N` true-valued cells.
             *     Broadcasting rules apply.
             * x : ndarray<T> const&
             *     Two possibilities:
             *     * (1,) value broadcasted to places where `mask` is :cpp:obj:`true`;
             *     * (N,) values to place where `mask` is :cpp:obj:`true`. (no broadcasting)
             */
            ndarray<T>& filter(ndarray<bool> const& mask,
                               ndarray<T> const& x);
            ndarray<T>& filter(ndarray<bool> const& mask,
                               T const& x);

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
             * Parameters
             * ----------
             * axes : std::vector<size_t> const&
             *     Subset of single-element dimensions to remove.
             *     If not provided, remove all single-element dimensions.
             *
             * Returns
             * -------
             * squeezed : ndarray<T>
             *     The input array with select dimensions of length 1 removed.
             *     This is always :cpp:ptr:`this` itself or a view into it.
             */
            ndarray<T> squeeze(std::vector<size_t> const& axes) const;
            ndarray<T> squeeze() const;

            /*
             * Returns an array containing the same data with a new shape.
             *
             * Returns
             * -------
             * reshaped : ndarray<T>
             *     This will be a new view object if possible; otherwise, it
             *     will be a copy.
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
             * Parameters
             * ----------
             * axes : std::vector<size_t> const&
             *     Reordered dimension sequence.
             *     If not provided, reverse axes.
             *
             * Returns
             * -------
             * tr : ndarray<T>
             *     View on :cpp:ptr:`this` with axes suitably permuted.
             */
            ndarray<T> transpose(std::vector<size_t> const& axes) const;
            ndarray<T> transpose() const;

            /*
             * Element-wise static_cast<U>() of the array.
             *
             * Returns
             * -------
             * casted : ndarray<U>
             *     Copy of :cpp:ptr:`this` with element-wise static_cast<U>()
             *     of the input.
             *
             * Notes
             * -----
             * This method might need to be called as such:
             *     auto x = nd::zeros<T1>({5, 3, 4});
             *     auto y = x.template cast<T2>();
             */
            template <typename U>
            ndarray<U> cast() const;

            /*
             * Copy RHS array contents into LHS array.
             * Broadcasting rules apply.
             */
            ndarray<T>& operator=(ndarray<T> const& other);
            ndarray<T>& operator=(T const other);

            ndarray<T>& operator+=(ndarray<T> const& other);
            ndarray<T>& operator+=(T const other);

            ndarray<T>& operator-=(ndarray<T> const& other);
            ndarray<T>& operator-=(T const other);

            ndarray<T>& operator*=(ndarray<T> const& other);
            ndarray<T>& operator*=(T const other);

            ndarray<T>& operator/=(ndarray<T> const& other);
            ndarray<T>& operator/=(T const other);

            ndarray<T>& operator%=(ndarray<T> const& other);
            ndarray<T>& operator%=(T const other);

            ndarray<T>& operator&=(ndarray<T> const& other);
            ndarray<T>& operator&=(T const other);

            ndarray<T>& operator|=(ndarray<T> const& other);
            ndarray<T>& operator|=(T const other);

            ndarray<T>& operator^=(ndarray<T> const& other);
            ndarray<T>& operator^=(T const other);

            ndarray<T>& operator<<=(ndarray<T> const& other);
            ndarray<T>& operator<<=(T const other);

            ndarray<T>& operator>>=(ndarray<T> const& other);
            ndarray<T>& operator>>=(T const other);

            ndarray<T>& operator++();
            ndarray<T>& operator--();

            ndarray<T> operator-() const;

            ndarray<T> operator~() const;

            ndarray<bool> operator!() const;

            ndarray<T> operator+(ndarray<T> const& other) const;
            ndarray<T> operator+(T const other) const;

            ndarray<T> operator-(ndarray<T> const& other) const;
            ndarray<T> operator-(T const other) const;

            ndarray<T> operator*(ndarray<T> const& other) const;
            ndarray<T> operator*(T const other) const;

            ndarray<T> operator/(ndarray<T> const& other) const;
            ndarray<T> operator/(T const other) const;

            ndarray<T> operator%(ndarray<T> const& other) const;
            ndarray<T> operator%(T const other) const;

            ndarray<T> operator&(ndarray<T> const& other) const;
            ndarray<T> operator&(T const other) const;

            ndarray<T> operator|(ndarray<T> const& other) const;
            ndarray<T> operator|(T const other) const;

            ndarray<T> operator^(ndarray<T> const& other) const;
            ndarray<T> operator^(T const other) const;

            ndarray<T> operator<<(ndarray<T> const& other) const;
            ndarray<T> operator<<(T const other) const;

            ndarray<T> operator>>(ndarray<T> const& other) const;
            ndarray<T> operator>>(T const other) const;

            ndarray<bool> operator&&(ndarray<bool> const& other) const;
            ndarray<bool> operator&&(bool const other) const;

            ndarray<bool> operator||(ndarray<bool> const& other) const;
            ndarray<bool> operator||(bool const other) const;

            ndarray<bool> operator==(ndarray<T> const& other) const;
            ndarray<bool> operator==(T const other) const;

            ndarray<bool> operator!=(ndarray<T> const& other) const;
            ndarray<bool> operator!=(T const other) const;

            ndarray<bool> operator<(ndarray<T> const& other) const;
            ndarray<bool> operator<(T const other) const;

            ndarray<bool> operator<=(ndarray<T> const& other) const;
            ndarray<bool> operator<=(T const other) const;

            ndarray<bool> operator>(ndarray<T> const& other) const;
            ndarray<bool> operator>(T const other) const;

            ndarray<bool> operator>=(ndarray<T> const& other) const;
            ndarray<bool> operator>=(T const other) const;
    };



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
            void advance();

        public:
            ndarray_iterator() = delete;
            ndarray_iterator(ndarray<T>* const x);
            ndarray_iterator(ndarray<T>* const x,
                             index_t const& index);
            ndarray_iterator(ndarray_iterator<T> const& other);
            ~ndarray_iterator();

            ndarray_iterator<T>& operator=(ndarray_iterator<T> const& other);
            bool operator==(ndarray_iterator<T> const& other) const;
            bool operator!=(ndarray_iterator<T> const& other) const;
            T& operator*() const;
            ndarray_iterator<T>& operator++();
    };
}

namespace nd {  // Functions
    /*
     * Returns
     * -------
     * pi : T
     *     Mathematical constant \pi \approx 3.1415
     */
    template <typename T>
    T constexpr pi();

    /*
     * Returns
     * -------
     * e : T
     *     Euler's constant e \approx 2.71828
     */
    template <typename T>
    T constexpr e();

    /*
     * Returns
     * -------
     * j : T
     *     Imaginary constant j = \sqrt(-1)
     */
    template <typename T>
    T constexpr j();

    /*
     * Return a contiguous array in memory (C-order).
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Input array.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     Contiguous array of same shape and content as `x`.
     *     If `x` was already contiguous, then a view is returned.
     */
    template <typename T>
    ndarray<T> ascontiguousarray(ndarray<T> const& x);

    /*
     * Create 1-D array from elements.
     *
     * Parameters
     * ----------
     * ilist : std::initializer_list<T>
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     1-D array containing elements from `ilist`.
     */
    template <typename T>
    ndarray<T> r_(std::initializer_list<T> ilist);

    /*
     * Evenly-spaced values in a given interval.
     *
     * Parameters
     * ----------
     * start : T const
     *     Start of the interval.
     *     The interval includes this value.
     * stop : T const
     *     End of interval.
     *     The interval does not include this value, except in some cases where
     *     step is not an integer and floating point round-off affects the length
     *     of `out`.
     * step : T const
     *     Spacing between values.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     1-D array of evenly-spaced values.
     *     For T=<floats>, the length of the result is `ceil((stop - start)/step)`.
     *     Because of floating point overflow, this rule may result in the last
     *     element being greater than `stop`.
     */
    template <typename T>
    ndarray<T> arange(T const start,
                      T const stop,
                      T const step);

    /*
     * Evenly spaced numbers over a specified interval.
     *
     * Parameters
     * ----------
     * start : T const
     *     Start of the interval.
     * stop : T const
     *     End of the interval.
     * N : size_t const
     *     Number of samples to generate.
     * endpoint : bool const
     *     If true (default), `stop` is the last sample.
     *     If false, `stop` is not included.
     *
     * Returns
     * -------
     * samples : ndarray<T>
     *     (N,) array with equally-spaced values.
     *     If `endpoint` is true,  samples[i] = start + i * ((stop - start) / (N - 1)).
     *     If `endpoint` is false, samples[i] = start + i * ((stop - start) / N).
     */
    template <typename T>
    ndarray<T> linspace(T const start,
                        T const stop,
                        size_t const N,
                        bool const endpoint = true);

    /*
     * (Sparse) coordinate arrays from coordinate vectors.
     *
     * Parameters
     * ----------
     * x : std::vector<ndarray<T>> const&
     *     1-D arrays representing coordinates of a grid.
     *
     * Returns
     * -------
     * mesh : std::vector<ndarray<T>>
     *     mesh[i] = x[i].reshape({1, ..., x[i].size(), ..., 1})
     */
    template <typename T>
    std::vector<ndarray<T>> meshgrid(std::vector<ndarray<T>> const& x);

    /*
     * Returns
     * -------
     * out : ndarray<T>
     *     Array of `value` of the given shape.
     */
    template <typename T>
    ndarray<T> full(shape_t const& shape,
                    T const value);

    /*
     * Returns
     * -------
     * out : ndarray<T>
     *     Array of zeros of the given shape.
     */
    template <typename T>
    ndarray<T> zeros(shape_t const& shape);

    /*
     * Returns
     * -------
     * out : ndarray<T>
     *     Array of ones of the given shape.
     */
    template <typename T>
    ndarray<T> ones(shape_t const& shape);

    /*
     * 2-D array with ones on the main diagonal and zeros elsewhere.
     *
     * Parameters
     * ----------
     * N : size_t const
     *     Number of rows.
     *
     * Returns
     * -------
     * I : ndarray<T>
     *     (N, N) array where all elements are 0, except on the main diagonal
     *     where they are 1.
     */
    template <typename T>
    ndarray<T> eye(size_t const N);

    /*
     * Logical OR along specified axis.
     *
     * Parameters
     * ----------
     * x : ndarray<bool> const&
     *     N-D input array.
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as dimension
     *      of size 1.
     * out : ndarray<bool>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<bool>
     *     OR-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T, typename BOOL = bool>
    ndarray<BOOL> any(ndarray<T> const& x,
                      size_t const axis,
                      bool const keepdims = false,
                      ndarray<T>* const out = nullptr);

    /*
     * Logical AND along specified axis.
     *
     * Parameters
     * ----------
     * x : ndarray<bool> const&
     *     N-D input array.
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<bool>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<bool>
     *     AND-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T, typename BOOL = bool>
    ndarray<BOOL> all(ndarray<T> const& x,
                      size_t const axis,
                      bool const keepdims = false,
                      ndarray<T>* const out = nullptr);

    /*
     * Element-wise closeness check.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * y : ndarray<T> const&
     * rtol : double const
     *     Relative tolerance for float comparisons.
     *     This parameter is meaningless for {bool, int} types.
     * atol : double const
     *     Absolute tolerance for float comparisons.
     *     This parameter is meaningless for {bool, int} types.
     * out : ndarray<bool>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * close_enough: ndarray<bool>
     *     (bool, int) -> x == y
     *     (float, complex) -> |x - y| <= (atol + rtol * |y|)
     */
    template <typename T, typename BOOL = bool>
    ndarray<BOOL> isclose(ndarray<T> const& x,
                          ndarray<T> const& y,
                          ndarray<bool>* const out = nullptr,
                          double const rtol = 1e-5,
                          double const atol = 1e-8);

    /*
     * Element-wise closeness check.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * y : ndarray<T> const&
     * rtol : double const
     *     Relative tolerance for float comparisons.
     *     This parameter is meaningless for {bool, int} types.
     * atol : double const
     *     Absolute tolerance for float comparisons.
     *     This parameter is meaningless for {bool, int} types.
     *
     * Returns
     * -------
     * all_close_enough: bool
     *     (bool, int) -> all(x == y)
     *     (float, complex) -> all(|x - y| <= (atol + rtol * |y|))
     */
    template <typename T, typename BOOL = bool>
    BOOL allclose(ndarray<T> const& x,
                  ndarray<T> const& y,
                  double const rtol = 1e-5,
                  double const atol = 1e-8);

    /*
     * Sum of array elements over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Elements to sum.
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     ADD-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> sum(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims = false,
                   ndarray<T>* const out = nullptr);

    /*
     * Product of array elements over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Elements to multiply.
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     TIMES-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> prod(ndarray<T> const& x,
                    size_t const axis,
                    bool const keepdims = false,
                    ndarray<T>* const out = nullptr);

    /*
     * Join a sequence of arrays along a new axis.
     *
     * Parameters
     * ----------
     * x : std::vector<ndarray<T>> const&
     *     Arrays to join.
     *     All arrays must have the same shape.
     * axis : size_t const
     *     Dimension in the output array along which arrays are joined.
     *     Must lie in {0, ..., x[0].ndim()}.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     Stacked array with one more dimension (at position=axis) than input arrays.
     */
    template <typename T>
    ndarray<T> stack(std::vector<ndarray<T>> const& x,
                     size_t const axis,
                     ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric sine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Angle [rad].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     sin(x)
     */
    template <typename T>
    ndarray<T> sin(ndarray<T> const& x,
                   ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric cosine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Angle [rad].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     cos(x)
     */
    template <typename T>
    ndarray<T> cos(ndarray<T> const& x,
                   ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric tangent.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Angle [rad].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     tan(x)
     */
    template <typename T>
    ndarray<T> tan(ndarray<T> const& x,
                   ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric inverse sine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Value in [-1, 1].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arcsin(x) \in [-\pi/2, \pi/2], NaN if out-of-range.
     */
    template <typename T>
    ndarray<T> arcsin(ndarray<T> const& x,
                      ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric inverse cosine.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Value in [-1, 1].
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arccos(x) \in [0, \pi], NaN if out-of-range.
     */
    template <typename T>
    ndarray<T> arccos(ndarray<T> const& x,
                      ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric inverse tangent.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arctan(x) \in [-\pi/2, \pi/2]
     */
    template <typename T>
    ndarray<T> arctan(ndarray<T> const& x,
                      ndarray<T>* const out = nullptr);

    /*
     * Element-wise trigonometric inverse tangent of x1 / x2, choosing the
     * quadrant correctly.
     *
     * Parameters
     * ----------
     * x1 : ndarray<T> const&
     * x2 : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     arctan(x1 / x2) \in [-\pi, \pi]
     */
    template <typename T>
    ndarray<T> arctan2(ndarray<T> const& x1,
                       ndarray<T> const& x2,
                       ndarray<T>* const out = nullptr);

    /*
     * Convert angles from degrees to radians.
     *
     * Parameters
     * ----------
     * deg : ndarray<T> const&
     *     Angle [deg]
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * rad : ndarray<T>
     *     Angle [rad]
     */
    template <typename T>
    ndarray<T> deg2rad(ndarray<T> const& deg,
                       ndarray<T>* const out = nullptr);

    /*
     * Convert angles from radians to degrees.
     *
     * Parameters
     * ----------
     * rad : ndarray<T> const&
     *     Angle [rad]
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * deg : ndarray<T>
     *     Angle [deg]
     */
    template <typename T>
    ndarray<T> rad2deg(ndarray<T> const& rad,
                       ndarray<T>* const out = nullptr);

    /*
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     sin(\pi x) / (\pi x)
     */
    template <typename T>
    ndarray<T> sinc(ndarray<T> const& x,
                    ndarray<T>* const out = nullptr);

    /*
     * Element-wise absolution value.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     abs(x)
     */
    template <typename T>
    ndarray<T> abs(ndarray<T> const& x,
                   ndarray<T>* const out = nullptr);

    /*
     * Element-wise base-E exponentiation.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     e ** x
     */
    template <typename T>
    ndarray<T> exp(ndarray<T> const& x,
                   ndarray<T>* const out = nullptr);

    /*
     * Element-wise base-E logarithm.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     ln(x) \in \bR, NaN if x < 0.
     */
    template <typename T>
    ndarray<T> log(ndarray<T> const& x,
                   ndarray<T>* const out = nullptr);

    /*
     * Find the unique elements of an array.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     *     Input array.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     (N,) array with sorted unique values of `x.ravel()`.
     */
    template <typename T>
    ndarray<T> unique(ndarray<T> const& x);

    /*
     * Standard deviation over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     STD-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     *
     * Notes
     * -----
     * STD is implemented as mean(abs(x - x.mean()) ** 2).
     */
    template <typename T>
    ndarray<T> std(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims = false,
                   ndarray<T>* const out = nullptr);

    /*
     * Average over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     MEAN-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> mean(ndarray<T> const& x,
                    size_t const axis,
                    bool const keepdims = false,
                    ndarray<T>* const out = nullptr);

    /*
     * Minimum of array elements over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     MIN-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> min(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims = false,
                   ndarray<T>* const out = nullptr);

    /*
     * Maximum of array elements over given axis.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * axis : size_t const
     *     Dimension along which to reduce.
     * keepdims : bool const
     *     If true, the axis which is reduced is left in the result as
     *     dimension of size 1.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     MAX-reduced array with
     *     * (x.ndim() - 1) dimensions if `keepdims` is false;
     *     * x.ndim() dimensions if `keepdims` is true.
     */
    template <typename T>
    ndarray<T> max(ndarray<T> const& x,
                   size_t const axis,
                   bool const keepdims = false,
                   ndarray<T>* const out = nullptr);

    /*
     * Element-wise ceiling.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     */
    template <typename T>
    ndarray<T> ceil(ndarray<T> const& x,
                    ndarray<T>* const out = nullptr);

    /*
     * Element-wise floor.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     */
    template <typename T>
    ndarray<T> floor(ndarray<T> const& x,
                     ndarray<T>* const out = nullptr);

    /*
     * Element-wise clip.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * down : T const
     * up : T const
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     Saturate `x` to lie in [down, up].
     */
    template <typename T>
    ndarray<T> clip(ndarray<T> const& x,
                    T const down,
                    T const up,
                    ndarray<T>* const out = nullptr);

    /*
     * Element-wise indication of number's sign.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * y : ndarray<T>
     *     -1 if x < 0, 0 if x==0, 1 if x > 0.
     */
    template <typename T>
    ndarray<T> sign(ndarray<T> const& x,
                    ndarray<T>* const out = nullptr);

    /*
     * Float-view into complex-valued data.
     *
     * Parameters
     * ----------
     * x : ndarray<std::complex<T>> const&
     *     (...) data
     *
     * Returns
     * -------
     * x_f : ndarray<T>
     *     (..., 2) view of `x` as floats.
     *     x_f[..., 0] = x.real
     *     x_f[..., 1] = x.imag
     */
    template <typename T>
    ndarray<T> asfloat(ndarray<std::complex<T>> const& x);

    /*
     * Element-wise real-part extraction.
     *
     * Parameters
     * ----------
     * x : ndarray<std::complex<T>> const&
     *
     * Returns
     * -------
     * out : ndarray<T>
     *     View on real-part of `x`.
     */
    template <typename T>
    ndarray<T> real(ndarray<std::complex<T>> const& x);

    /*
     * Element-wise imag-part extraction.
     *
     * Parameters
     * ----------
     * x : ndarray<std::complex<T>> const&
     *
     * Returns
     * -------
     * out : ndarray<T>
     *     View on imag-part of `x`.
     */
    template <typename T>
    ndarray<T> imag(ndarray<std::complex<T>> const& x);

    /*
     * Element-wise conjugation.
     *
     * Parameters
     * ----------
     * x : ndarray<T> const&
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the input.
     *
     * Returns
     * -------
     * out : ndarray<T>
     *     conj(x)
     */
    template <typename T>
    ndarray<T> conj(ndarray<T> const& x,
                    ndarray<T>* const out = nullptr);
}

namespace nd::linalg {  // Linear Algebra routines
    /*
     * Matrix-Multiplication extended to ND-arrays.
     *
     * Parameters
     * ----------
     * A : ndarray<T> const&
     *     (a[0], ..., a[N-2], d) array.
     * B : ndarray<T> const&
     *     (d, b[1], ..., b[M-1]) array.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output, and be contiguous.
     *
     * Returns
     * -------
     * C : ndarray<T>
     *     (a[0], ..., a[N-2], b[1], ..., b[M-1]) array.
     *
     * Examples
     * --------
     * Let A \in \bR^{5, 3, 4} and B \in \bR^{4, 7, 2}.
     * Then C = mm(A, B) \in \bR^{5, 3, 7, 2} such that
     *     C[i,j,k,l] = \sum_{q = 0}^{3} A[i,j,q] B[q,k,l]
     *
     * Let A \in \bR^{5} and B \in \bR^{5}.
     * Then C = mm(A, B) \in \bR^{1} such that
     *     C[0] = \sum_{q = 0}^{4} A[q] B[q]
     */
    template <typename T>
    ndarray<T> mm(ndarray<T> const& A,
                  ndarray<T> const& B,
                  ndarray<T>* const out = nullptr);

    /*
     * Batch Matrix-Multiplication.
     *
     * Parameters
     * ----------
     * A : ndarray<T> const&
     *     ([M,] N, P) array.
     * B : ndarray<T> const&
     *     ([M,] P, Q) array.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * C : ndarray<T>
     *     (M, N, Q) layer-wise matrix product of `A` and `B`.
     *     Broadcasting rules apply along upper dimensions.
     *
     * Examples
     * --------
     * Let A \in \bR^{M, N, P} and B \in \bR^{M, P, Q}.
     * Then C = bmm(A, B) \in \bR^{M, N, Q} such that
     *     C[i, :, :] = mm(A[i, :, :], B[i, :, :])
     *
     * Let A \in \bR^{M, N, P} and B \in \bR^{P, Q}.
     * Then C = bmm(A, B) \in \bR^{M, N, Q} such that
     *     C[i, :, :] = mm(A[i, :, :], B[:, :])
     *
     * Notes
     * -----
     * This is a convenience function that calls nd::linalg::mm() under the hood.
     */
    template <typename T>
    ndarray<T> bmm(ndarray<T> const& A,
                   ndarray<T> const& B,
                   ndarray<T>* const out = nullptr);
}

namespace {  // IO
    /*
     * Send a (numeric) vector to output stream.
     */
    template <typename T>
    std::ostream& operator<<(std::ostream& os,
                             std::vector<T> const& v);

    /*
     * Send ndarray to output stream.
     *
     * Only 1d/2d arrays can be printed.
     */
    template <typename T>
    std::ostream& operator<<(std::ostream& os,
                             nd::ndarray<T> const& x);
}

#endif // NDFORWARD_HPP
