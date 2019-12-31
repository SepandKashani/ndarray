// ############################################################################
// _ndcontainer.hpp
// ================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDCONTAINER_HPP
#define _NDCONTAINER_HPP

#include <cstdint>

#include "_ndtype.hpp"

namespace nd::util {
    template <typename T> void NDARRAY_ASSERT(bool const cond, T const& msg);
}

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

            ndcontainer(size_t const nbytes):
                m_buffer(new byte_t[nbytes + byte_alignment - 1]),
                m_size(nbytes),
                m_own_memory(true) {
                    uintptr_t const mask_lsb = reinterpret_cast<uintptr_t>(byte_alignment - 1);
                    uintptr_t const mask_msb = ~mask_lsb;
                    uintptr_t const unaligned_start = reinterpret_cast<uintptr_t>(m_buffer);

                    if((mask_lsb & unaligned_start) == 0) {
                        m_offset = 0;
                    } else {
                        uintptr_t const aligned_start = (unaligned_start & mask_msb) + byte_alignment;
                        m_offset = reinterpret_cast<size_t>(aligned_start - unaligned_start);
                    }

                    util::NDARRAY_ASSERT(m_offset <= byte_alignment - 1, "Alignment issue.");
                }

            ndcontainer(byte_t* const data, size_t const nbytes):
                m_buffer(data),
                m_size(nbytes),
                m_own_memory(false),
                m_offset(0) {}

            ~ndcontainer() {
                if(m_own_memory) {
                    delete[] m_buffer;
                }
            }

            byte_t* data() const { return m_buffer + m_offset; }

            size_t nbytes() const { return m_size; }

            bool own_memory() const { return m_own_memory; }

            void operator=(ndcontainer const& other) = delete;
    };
}

#endif // _NDCONTAINER_HPP
