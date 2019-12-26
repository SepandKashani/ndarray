// ############################################################################
// test_type.hpp
// =============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_TYPE_HPP
#define TEST_TYPE_HPP

#include <complex>

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    // Elementary Types
    typedef ::testing::Types<bool> MyBoolTypes;
    typedef ::testing::Types<int> MySignedIntTypes;
    typedef ::testing::Types<int, size_t> MyIntTypes;
    typedef ::testing::Types<float, double> MyFloatTypes;
    typedef ::testing::Types<cfloat, cdouble> MyComplexTypes;

    // Combination Types
    typedef ::testing::Types<bool, int, size_t> MyBoolIntTypes;
    typedef ::testing::Types<int, float, double> MySignedIntFloatTypes;
    typedef ::testing::Types<int, size_t, float, double> MyIntFloatTypes;
    typedef ::testing::Types<int,
                             float, double,
                             cfloat, cdouble> MySignedIntFloatComplexTypes;
    typedef ::testing::Types<int, size_t,
                             float, double,
                             cfloat, cdouble> MyIntFloatComplexTypes;
    typedef ::testing::Types<float, double,
                             cfloat, cdouble> MyFloatComplexTypes;
    typedef ::testing::Types<bool, int, size_t,
                             float, double,
                             cfloat, cdouble> MyArithmeticTypes;
}

#endif // TEST_TYPE_HPP
