// ############################################################################
// test_ndfunc.cpp
// ===============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDFUNC_CPP
#define TEST_NDFUNC_CPP

#include <cmath>
#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T>
    class TestNdFunc : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFunc);

    TYPED_TEST_P(TestNdFunc, TestPi) {
        TypeParam const tested = pi<TypeParam>();
        TypeParam const correct = static_cast<TypeParam>(M_PI);

        ASSERT_EQ(tested, correct);
    }

    TYPED_TEST_P(TestNdFunc, TestEuler) {
        TypeParam const tested = e<TypeParam>();
        TypeParam const correct = static_cast<TypeParam>(M_E);

        ASSERT_EQ(tested, correct);
    }

    typedef ::testing::Types<float, double> MyTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFunc,
                               TestPi,
                               TestEuler);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFunc, MyTypes);
}

#endif // TEST_NDFUNC_CPP
