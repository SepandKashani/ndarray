// ############################################################################
// _ndcontainer.hpp
// ================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDCONTAINER_HPP
#define _NDCONTAINER_HPP

#include "_ndtype.hpp"

namespace nd {
    /*
     * Thin shell around a memory buffer.
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
            ndcontainer() = delete;
            ndcontainer(ndcontainer const& other) = delete;
            ndcontainer(ndcontainer const&& other) = delete;
            ndcontainer(size_t const nbytes);
            ndcontainer(byte_t* const data, size_t const nbytes);
            ~ndcontainer();

            byte_t* data() const;
            size_t nbytes() const;
            bool own_memory() const;

            void operator=(ndcontainer const& other) = delete;
    };
}

#endif // _NDCONTAINER_HPP
