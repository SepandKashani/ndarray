// ############################################################################
// _ndvector.hpp
// =============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDVECTOR_HPP
#define _NDVECTOR_HPP

#include <algorithm>
#include <array>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>

#include <iostream>

namespace nd::util {
    template <typename T> void NDARRAY_ASSERT(bool const cond, T const& msg);
}

namespace nd {
    /*
     * STL vector-like container backed by stack-allocated array.
     *
     * Why does this class exist?
     * --------------------------
     * Originally (nd::shape_t, nd::stride_t) were implemented with std::vector.
     *
     * The way nd::ndarray(s) are handled however means that they are (very)
     * frequently shallow-copied around the place. The consequence is that
     * dynamic memory allocation is omnipresent when manipulating these data
     * structures.
     *
     * In practice most arrays created are of small rank, say < 8. Knowing this,
     * it is worthwhile to stack-allocate these structures with some extra space.
     *
     * Note
     * ----
     * * To guarantee backwards compatibility with previously-written code
     *   using nd::ndarray, we mimic useful parts of std::vector's public
     *   interface. Methods that grow the array will throw errors at runtime
     *   if total capacity exceeds N_MAX.
     *
     * * N_MAX is chosen to cater to the needs of most users. Should it be
     *   required to create arrays with more dimensions, increase N_MAX.
     */
    template <typename T, size_t N_MAX = 8>
    class vector {
        static_assert(N_MAX > 0, "N_MAX must be positive.");

        public:
            typedef typename std::array<T, N_MAX>::value_type      value_type;
            typedef typename std::array<T, N_MAX>::size_type       size_type;
            typedef typename std::array<T, N_MAX>::difference_type difference_type;
            typedef value_type&                                    reference;
            typedef value_type const&                              const_reference;
            typedef value_type*                                    pointer;
            typedef value_type* const                              const_pointer;
            typedef typename std::array<T, N_MAX>::iterator        iterator;
            typedef typename std::array<T, N_MAX>::const_iterator  const_iterator;
            typedef std::reverse_iterator<iterator>                reverse_iterator;
            typedef std::reverse_iterator<const_iterator>          const_reverse_iterator;

        private:
            size_type m_size = 0;
            std::array<T, N_MAX> m_data;

            #define NDVECTOR_ERROR_CAPACITY "At most N_MAX elements allowed."

            /*
             * Binary AND-fold (valid) elements of input buffers.
             *
             * Parameters
             * ----------
             * ufunc : F
             *     Binary function to apply.
             *     Must have signature "bool f(T const&, T const&)".
             * other : vector<T, N_MAX> const&
             *     (N,) input buffer (rhs).
             *
             * Returns
             * -------
             * tf : bool
             *
             * Notes
             * -----
             * Two vectors can be folded only if they have the same size N.
             * The result will always be `false` if not satisfied.
             */
            template <typename F>
            bool fold(F ufunc, vector<T, N_MAX> const& other) const {
                bool tf = false;
                if(size() == other.size()) {
                    using vt = vector<T, N_MAX>;

                    vt* _this = const_cast<vt*>(this);
                    vt& _other = const_cast<vt&>(other);
                    tf = std::inner_product(_this->begin(), _this->end(),
                                            _other.begin(), true,
                                            std::logical_and<bool>(),
                                            ufunc);
                }
                return tf;
            }

        public:
            vector():
                m_size(0) {}

            vector(size_type count, T const& value):
                m_size(count) {
                    util::NDARRAY_ASSERT(count <= N_MAX, NDVECTOR_ERROR_CAPACITY);
                    std::fill_n(m_data.begin(), m_size, value);
                }

            explicit vector(size_type count):
                m_size(count) {
                    util::NDARRAY_ASSERT(count <= N_MAX, NDVECTOR_ERROR_CAPACITY);
                }

            template <typename InputIt>
            vector(InputIt first, InputIt last) {
                auto const& d = std::distance(first, last);
                auto const d_u = static_cast<size_type>(d);
                util::NDARRAY_ASSERT((0 <= d) && (d_u <= N_MAX), NDVECTOR_ERROR_CAPACITY);

                m_size = d_u;
                std::copy(first, last, m_data.begin());
            }

            vector(vector<T, N_MAX> const& other):
                m_size(other.m_size),
                m_data(other.m_data) {}

            vector(std::initializer_list<T> init):
                m_size(init.size()) {
                    util::NDARRAY_ASSERT(init.size() <= N_MAX, NDVECTOR_ERROR_CAPACITY);
                    std::copy_n(init.begin(), m_size, m_data.begin());
                }

            ~vector() {}

            vector<T, N_MAX>& operator=(vector<T, N_MAX> const& other) {
                m_size = other.m_size;
                m_data = other.m_data;
                return *this;
            }

            vector<T, N_MAX>& operator=(std::initializer_list<T> ilist) {
                util::NDARRAY_ASSERT(ilist.size() <= N_MAX, NDVECTOR_ERROR_CAPACITY);
                m_size = ilist.size();
                std::copy_n(ilist.begin(), m_size, m_data.begin());
                return *this;
            }

            reference operator[](size_type pos) {
                return m_data[pos];
            }

            T* data() noexcept {
                return m_data.data();
            }

            iterator begin() noexcept {
                return m_data.begin();
            }

            iterator end() noexcept {
                return m_data.begin() + m_size;
            }

            reverse_iterator rbegin() noexcept {
                return reverse_iterator(end());
            }

            reverse_iterator rend() noexcept {
                return reverse_iterator(begin());
            }

            bool empty() const noexcept {
                return m_size == static_cast<size_type>(0);
            }

            size_type size() const noexcept {
                return m_size;
            }

            size_type max_size() const noexcept {
                return N_MAX;
            }

            void clear() noexcept {
                m_size = 0;
            }

            iterator insert(iterator pos, T const& value) {
                util::NDARRAY_ASSERT(m_size < N_MAX, NDVECTOR_ERROR_CAPACITY);
                util::NDARRAY_ASSERT((begin() <= pos) && (pos <= end()),
                                     "Parameter[pos] is out of bounds.");

                size_type const d = std::distance(pos, end());
                std::copy_n(rbegin(), d, rbegin() - 1);
                *pos = value;
                m_size += 1u;

                return pos;
            }

            void push_back(T const& value) {
                util::NDARRAY_ASSERT(m_size < N_MAX, NDVECTOR_ERROR_CAPACITY);
                m_data[m_size] = value;
                m_size += 1;
            }

            void pop_back() {
                if(m_size > 0) {
                    m_size -= 1;
                }
            }

            bool operator==(vector<T, N_MAX> const& other) {
                return fold(std::equal_to<T>(), other);
            }

            bool operator!=(vector<T, N_MAX> const& other) {
                return !operator==(other);
            }
    };
}

#endif // _NDVECTOR_HPP
