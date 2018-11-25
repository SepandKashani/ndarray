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
    /* Constants =========================================================== */
    template <typename T>
    class TestNdFuncConstant : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncConstant);

    TYPED_TEST_P(TestNdFuncConstant, TestPi) {
        TypeParam const tested = pi<TypeParam>();
        TypeParam const correct = static_cast<TypeParam>(M_PI);

        ASSERT_EQ(tested, correct);
    }

    TYPED_TEST_P(TestNdFuncConstant, TestEuler) {
        TypeParam const tested = e<TypeParam>();
        TypeParam const correct = static_cast<TypeParam>(M_E);

        ASSERT_EQ(tested, correct);
    }

    typedef ::testing::Types<float, double> MyNdFuncConstantTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncConstant,
                               TestPi,
                               TestEuler);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncConstant, MyNdFuncConstantTypes);


    /* Utility ============================================================= */
    template <typename T>
    class TestNdFuncUtility : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncUtility);
    TYPED_TEST_P(TestNdFuncUtility, TestRUnderscore) {
       {std::vector<TypeParam> const x;
        ndarray<TypeParam> y = r_(x);
        ASSERT_EQ(y.shape(), shape_t({x.size()}));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& tested_elem = y[{i}];
            TypeParam const& correct_elem = x[i];
            ASSERT_EQ(tested_elem, correct_elem);
        }}

       {std::vector<TypeParam> const x {static_cast<TypeParam>(1),
                                        static_cast<TypeParam>(2),
                                        static_cast<TypeParam>(3)};
        ndarray<TypeParam> y = r_(x);
        ASSERT_EQ(y.shape(), shape_t({x.size()}));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& tested_elem = y[{i}];
            TypeParam const& correct_elem = x[i];
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdFuncUtility, TestZeros) {
        ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, static_cast<TypeParam>(0.0));
        }
    }

    TYPED_TEST_P(TestNdFuncUtility, TestOnes) {
        ndarray<TypeParam> x = ones<TypeParam>({5, 3, 4});
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, static_cast<TypeParam>(1.0));
        }
    }

    TYPED_TEST_P(TestNdFuncUtility, TestFull) {
        TypeParam value = static_cast<TypeParam>(3.0);
        ndarray<TypeParam> x = full<TypeParam>({5, 3, 4}, value);
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, value);
        }
    }

    TYPED_TEST_P(TestNdFuncUtility, TestEye) {
        size_t const N = 5;
        ndarray<TypeParam> I = eye<TypeParam>(N);

        ASSERT_EQ(I.shape(), shape_t({N, N}));
        for(size_t i = 0; i < N; ++i) {
            for(size_t j = 0; j < N; ++j) {
                TypeParam const& tested_elem = I[{i, j}];
                TypeParam const correct_elem = ((i == j) ? static_cast<TypeParam>(1.0) : static_cast<TypeParam>(0.0));
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }
    }

    typedef ::testing::Types<bool, int, size_t,
                             float, double,
                             std::complex<float>,
                             std::complex<double>> MyNdFuncUtilityTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncUtility,
                               TestRUnderscore,
                               TestZeros,
                               TestOnes,
                               TestFull,
                               TestEye);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncUtility, MyNdFuncUtilityTypes);
}

#endif // TEST_NDFUNC_CPP
