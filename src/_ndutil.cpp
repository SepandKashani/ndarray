// ############################################################################
// _ndutil.cpp
// ===========
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDUTIL_CPP
#define _NDUTIL_CPP

#include <algorithm>
#include <ostream>
#include <sstream>
#include <stdexcept>

#include "_ndtype.hpp"
#include "_ndutil.hpp"

namespace nd { namespace util {
    template <typename T>
    void NDARRAY_ASSERT(bool const cond, T const& msg) {
        if(!cond) {
            throw std::runtime_error(msg);
        }
    }

    inline std::ostream& operator<<(std::ostream& os, shape_t const& shape) {
        os << "{";
        for(size_t i = 0; i < shape.size() - 1; ++i) {
            os << shape[i] << ", ";
        }
        os << shape[shape.size() - 1] << "}";

        return os;
    }

    inline shape_t predict_shape(shape_t const& lhs, shape_t const& rhs) {
        std::stringstream error_msg;
        error_msg << "Operands could not be broadcast together with shapes "
                  << "(" << lhs << ", " << rhs << ").\n";

        size_t const ndim_ohs = std::max<size_t>(lhs.size(), rhs.size());
        shape_t lhs_bcast(ndim_ohs, 1);
        shape_t rhs_bcast(ndim_ohs, 1);
        std::copy_n(lhs.rbegin(), lhs.size(), lhs_bcast.rbegin());
        std::copy_n(rhs.rbegin(), rhs.size(), rhs_bcast.rbegin());

        shape_t out(ndim_ohs, 1);
        for(size_t i = 0; i < out.size(); ++i) {
            size_t const& _lhs = lhs_bcast[i];
            size_t const& _rhs = rhs_bcast[i];
            size_t&       _out = out[i];

            NDARRAY_ASSERT((_lhs == _rhs) || (_lhs == 1) || (_rhs == 1),
                           error_msg.str());
            _out = std::max<size_t>(_lhs, _rhs);
        }

        return out;
    }

    inline slice::slice() {}

    inline slice::slice(int const stop):
        m_start {0},
        m_stop {stop},
        m_step {1} {}

    inline slice::slice(int const start, int const stop):
        m_start {start},
        m_stop {stop},
        m_step {1} {}

    inline slice::slice(int const start, int const stop, int const step):
        m_start {start},
        m_stop {stop},
        m_step {step} {
            NDARRAY_ASSERT(step != 0, "Zero steps are not allowed.");
        }

    inline slice::~slice() {}

    inline int slice::start() const {
        return m_start;
    }

    inline int slice::stop() const {
        return m_stop;
    }

    inline int slice::step() const {
        return m_step;
    }

    inline slice slice::map_limits(size_t const length) const {
        if(((m_start > m_stop) && (m_step > 0)) ||
           ((m_stop > m_start) && (m_step < 0))) {
            return slice(0, 0, m_step);
        }

        if(m_step > 0) {
            return slice(std::max<int>(0, m_start),
                         std::min<int>(m_stop, length),
                         m_step);
        } else {
            return slice(std::min<int>(m_start, length - 1),
                         std::max<int>(-1, m_stop),
                         m_step);
        }
    }
}}
#endif // _NDUTIL_CPP
