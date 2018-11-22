// ############################################################################
// test_ndarray_linalg.cpp
// =======================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_LINALG_CPP
#define TEST_NDARRAY_LINALG_CPP

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd { namespace linalg {
    TEST(TestBogus, TestBogusMatrixMultiply) {
        using T = float;
        size_t N(5), M(3), K(4);
        ndarray<T> A = zeros<T>({N, M, K});
        ndarray<T> B = ones<T>({N, M, K});
        ndarray<T> C = A + B;

        for(size_t i = 0; i < N; ++i) {
            for(size_t j = 0; j < M; ++j) {
                for(size_t k = 0; k < K; ++k) {
                    T const tested_elem = C[{i,j,k}];
                    T const correct_elem = A[{i,j,k}] + B[{i,j,k}];
                    ASSERT_EQ(tested_elem, correct_elem);
                }
            }
        }
    }
}}

#endif // TEST_NDARRAY_LINALG_CPP
