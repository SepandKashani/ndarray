// ############################################################################
// test_type.hpp
// =============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_TYPE_HPP
#define TEST_TYPE_HPP

#include <complex>

#include <gtest/gtest.h>

namespace nd {
    // Elementary Types
    typedef ::testing::Types<bool> MyBoolTypes;
    typedef ::testing::Types<int> MySignedIntTypes;
    typedef ::testing::Types<int, size_t> MyIntTypes;
    typedef ::testing::Types<float, double> MyFloatTypes;
    typedef ::testing::Types<std::complex<float>, std::complex<double>> MyComplexTypes;

    // Combination Types
    typedef ::testing::Types<bool, int, size_t> MyBoolIntTypes;
    typedef ::testing::Types<int, float, double> MySignedIntFloatTypes;
    typedef ::testing::Types<int, size_t, float, double> MyIntFloatTypes;
    typedef ::testing::Types<int,
                             float, double,
                             std::complex<float>, std::complex<double>> MySignedIntFloatComplexTypes;
    typedef ::testing::Types<int, size_t,
                             float, double,
                             std::complex<float>, std::complex<double>> MyIntFloatComplexTypes;
    typedef ::testing::Types<float, double,
                             std::complex<float>, std::complex<double>> MyFloatComplexTypes;
    typedef ::testing::Types<bool, int, size_t,
                             float, double,
                             std::complex<float>, std::complex<double>> MyArithmeticTypes;
}

#endif // TEST_TYPE_HPP
