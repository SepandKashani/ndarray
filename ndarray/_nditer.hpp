// ############################################################################
// _nditer.hpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDITER_HPP
#define _NDITER_HPP

#include <vector>

namespace nd {
    template <typename T> class ndarray;

    template <typename T>
    class ndarray_iterator {
        private:
            ndarray<T>*         m_iterable = nullptr;
            std::vector<size_t> m_index;
            int                 m_offset = 0;

            void advance();

        public:
            ndarray_iterator() = delete;
            ndarray_iterator(ndarray<T>* const x);
            ndarray_iterator(ndarray<T>* const x, std::vector<size_t> index);
            ndarray_iterator(ndarray_iterator<T> const& other);
            ~ndarray_iterator();

            ndarray_iterator<T>& operator=(ndarray_iterator<T> const& other);
            bool operator==(ndarray_iterator<T> const& other) const;
            bool operator!=(ndarray_iterator<T> const& other) const;
            T& operator*() const;
            ndarray_iterator<T>& operator++();
    };
}

#endif // _NDITER_HPP
