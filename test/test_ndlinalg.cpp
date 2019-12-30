// ############################################################################
// test_ndlinalg.cpp
// =======================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDLINALG_CPP
#define TEST_NDLINALG_CPP

#include <stdexcept>

#include <gtest/gtest.h>
#include "test_type.hpp"

#include "ndarray/ndarray.hpp"

namespace nd::linalg {
    template <typename T> class TestNdLinalgIntFloatComplex : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdLinalgIntFloatComplex);

    TYPED_TEST_P(TestNdLinalgIntFloatComplex, TestMM) {
        using ndT = ndarray<TypeParam>;

        // Valid shapes
        ASSERT_NO_THROW(mm(ndT(shape_t({1,})),      ndT(shape_t({1,}))));
        ASSERT_NO_THROW(mm(ndT(shape_t({3,})),      ndT(shape_t({3,}))));
        ASSERT_NO_THROW(mm(ndT(shape_t({2, 3})),    ndT(shape_t({3,}))));
        ASSERT_NO_THROW(mm(ndT(shape_t({4, 2, 3})), ndT(shape_t({3,}))));
        ASSERT_NO_THROW(mm(ndT(shape_t({2, 3})),    ndT(shape_t({3, 3}))));
        ASSERT_NO_THROW(mm(ndT(shape_t({1, 2, 3})), ndT(shape_t({3, 1, 3}))));
        ASSERT_NO_THROW(mm(ndT(shape_t({3,})),      ndT(shape_t({3, 2}))));

        // Wrong shapes
        ASSERT_THROW(mm(ndT(shape_t({2,})),   ndT(shape_t({1,}))), std::runtime_error);
        ASSERT_THROW(mm(ndT(shape_t({1,})),   ndT(shape_t({2,}))), std::runtime_error);
        ASSERT_THROW(mm(ndT(shape_t({3, 3})), ndT(shape_t({2, 4}))), std::runtime_error);

        // Valid output
        {   // (1,) x (1,) -> (1,)
            auto A = r_<int>({5,}).template cast<TypeParam>();
            auto B = r_<int>({3,}).template cast<TypeParam>();
            auto C_gt = r_<int>({15,}).template cast<TypeParam>();
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({1,}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (3,) x (3,) -> (1,)
            auto A = r_<int>({5, 3, 4}).template cast<TypeParam>();
            auto B = r_<int>({3, 1, 2}).template cast<TypeParam>();
            auto C_gt = r_<int>({26,}).template cast<TypeParam>();
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({1,}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (2, 3) x (3,) -> (2,)
            auto A = (r_<int>({1, 2, 3,
                               4, 5, 6})
                      .reshape({2, 3})
                      .template cast<TypeParam>());
            auto B = r_<int>({5, 7, 2}).template cast<TypeParam>();
            auto C_gt = r_<int>({25, 67}).template cast<TypeParam>();
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({2,}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (4, 2, 3) x (3,) -> (4, 2)
            auto A = ((arange<int>(0, 4 * 2 * 3, 1) + 1)
                      .reshape({4, 2, 3})
                      .template cast<TypeParam>());
            auto B = r_<int>({-5, 7, -2}).template cast<TypeParam>();
            auto C_gt = (r_<int>({3, 3,
                                  3, 3,
                                  3, 3,
                                  3, 3})
                         .reshape({4, 2})
                         .template cast<TypeParam>());
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({4, 2}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (2, 3) x (3, 3) -> (2, 3)
            auto A = ((arange<int>(0, 2 * 3, 1) + 1)
                      .reshape({2, 3})
                      .template cast<TypeParam>());
            auto B = ((arange<int>(0, 3 * 3, 1) + 5)
                      .reshape({3, 3})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({ 54,  60,  66,
                                  126, 141, 156})
                         .reshape({2, 3})
                         .template cast<TypeParam>());
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({2, 3}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (1, 2, 3) x (3, 1, 3) -> (1, 2, 1, 3)
            auto A = ((arange<int>(0, 2 * 3, 1) + 1)
                      .reshape({1, 2, 3})
                      .template cast<TypeParam>());
            auto B = ((arange<int>(0, 3 * 3, 1) + 5)
                      .reshape({3, 1, 3})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({ 54,  60,  66,
                                  126, 141, 156})
                         .reshape({1, 2, 1, 3})
                         .template cast<TypeParam>());
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({1, 2, 1, 3}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (3,) x (3, 2) -> (2,)
            auto A = (r_<int>({5, 3, 4})
                      .template cast<TypeParam>());
            auto B = (r_<int>({1, 2,
                               0, 5,
                               3, 7})
                      .reshape({3, 2})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({17, 53})
                         .template cast<TypeParam>());
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({2,}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }

        // Valid output (buffer provided)
        {   // (1,) x (1,) -> (1,)
            auto A = r_<int>({5,}).template cast<TypeParam>();
            auto B = r_<int>({3,}).template cast<TypeParam>();
            auto C_gt = r_<int>({15,}).template cast<TypeParam>();
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = mm(A, B, &C);
            ASSERT_TRUE(Z.equals(C));
            ASSERT_EQ(C.shape(), shape_t({1,}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (3,) x (3,) -> (1,)
            auto A = r_<int>({5, 3, 4}).template cast<TypeParam>();
            auto B = r_<int>({3, 1, 2}).template cast<TypeParam>();
            auto C_gt = r_<int>({26,}).template cast<TypeParam>();
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = mm(A, B, &C);
            ASSERT_TRUE(Z.equals(C));
            ASSERT_EQ(C.shape(), shape_t({1,}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (2, 3) x (3,) -> (2,)
            auto A = (r_<int>({1, 2, 3,
                               4, 5, 6})
                      .reshape({2, 3})
                      .template cast<TypeParam>());
            auto B = r_<int>({5, 7, 2}).template cast<TypeParam>();
            auto C_gt = r_<int>({25, 67}).template cast<TypeParam>();
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = mm(A, B, &C);
            ASSERT_TRUE(Z.equals(C));
            ASSERT_EQ(C.shape(), shape_t({2,}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (4, 2, 3) x (3,) -> (4, 2)
            auto A = ((arange<int>(0, 4 * 2 * 3, 1) + 1)
                      .reshape({4, 2, 3})
                      .template cast<TypeParam>());
            auto B = r_<int>({-5, 7, -2}).template cast<TypeParam>();
            auto C_gt = (r_<int>({3, 3,
                                  3, 3,
                                  3, 3,
                                  3, 3})
                         .reshape({4, 2})
                         .template cast<TypeParam>());
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = mm(A, B, &C);
            ASSERT_TRUE(Z.equals(C));
            ASSERT_EQ(C.shape(), shape_t({4, 2}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (2, 3) x (3, 3) -> (2, 3)
            auto A = ((arange<int>(0, 2 * 3, 1) + 1)
                      .reshape({2, 3})
                      .template cast<TypeParam>());
            auto B = ((arange<int>(0, 3 * 3, 1) + 5)
                      .reshape({3, 3})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({ 54,  60,  66,
                                  126, 141, 156})
                         .reshape({2, 3})
                         .template cast<TypeParam>());
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = mm(A, B, &C);
            ASSERT_TRUE(Z.equals(C));
            ASSERT_EQ(C.shape(), shape_t({2, 3}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (1, 2, 3) x (3, 1, 3) -> (1, 2, 1, 3)
            auto A = ((arange<int>(0, 2 * 3, 1) + 1)
                      .reshape({1, 2, 3})
                      .template cast<TypeParam>());
            auto B = ((arange<int>(0, 3 * 3, 1) + 5)
                      .reshape({3, 1, 3})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({ 54,  60,  66,
                                  126, 141, 156})
                         .reshape({1, 2, 1, 3})
                         .template cast<TypeParam>());
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = mm(A, B, &C);
            ASSERT_TRUE(Z.equals(C));
            ASSERT_EQ(C.shape(), shape_t({1, 2, 1, 3}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
        {   // (3,) x (3, 2) -> (2,)
            auto A = (r_<int>({5, 3, 4})
                      .template cast<TypeParam>());
            auto B = (r_<int>({1, 2,
                               0, 5,
                               3, 7})
                      .reshape({3, 2})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({17, 53})
                         .template cast<TypeParam>());
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = mm(A, B, &C);
            ASSERT_TRUE(Z.equals(C));
            ASSERT_EQ(C.shape(), shape_t({2,}));
            if constexpr (is_int<TypeParam>()) {
                ASSERT_TRUE(all((C == C_gt).ravel(), 0)[{0}]);
            } else {
                ASSERT_TRUE(allclose(C, C_gt));
            }
        }
    }

    // Initialize tests ====================================================
    REGISTER_TYPED_TEST_CASE_P(TestNdLinalgIntFloatComplex,
                               TestMM);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdLinalgIntFloatComplex, MyIntFloatComplexTypes);
}

#endif // TEST_NDLINALG_CPP
