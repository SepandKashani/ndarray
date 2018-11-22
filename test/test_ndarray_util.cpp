// ############################################################################
// test_ndarray_util.cpp
// =====================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_UTIL_CPP
#define TEST_NDARRAY_UTIL_CPP

#include <vector>

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T>
    class TestNdArrayUtil : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayUtil);

    TYPED_TEST_P(TestNdArrayUtil, TestRUnderscore) {
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

    TYPED_TEST_P(TestNdArrayUtil, TestZeros) {
        ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, static_cast<TypeParam>(0.0));
        }
    }

    TYPED_TEST_P(TestNdArrayUtil, TestOnes) {
        ndarray<TypeParam> x = ones<TypeParam>({5, 3, 4});
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, static_cast<TypeParam>(1.0));
        }
    }

    TYPED_TEST_P(TestNdArrayUtil, TestFull) {
        TypeParam value = static_cast<TypeParam>(3.0);
        ndarray<TypeParam> x = full<TypeParam>({5, 3, 4}, value);
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, value);
        }
    }

    TYPED_TEST_P(TestNdArrayUtil, TestEye) {
        size_t const N = 5;
        ndarray<TypeParam> I = eye<TypeParam>(N, N);

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
                             std::complex<double>> MyTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayUtil,
                               TestRUnderscore,
                               TestZeros,
                               TestOnes,
                               TestFull,
                               TestEye);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayUtil, MyTypes);
}

#endif // TEST_NDARRAY_UTIL_CPP
