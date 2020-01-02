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
        ASSERT_NO_THROW(mm(ndT({1,}),      ndT({1,})));
        ASSERT_NO_THROW(mm(ndT({3,}),      ndT({3,})));
        ASSERT_NO_THROW(mm(ndT({2, 3}),    ndT({3,})));
        ASSERT_NO_THROW(mm(ndT({4, 2, 3}), ndT({3,})));
        ASSERT_NO_THROW(mm(ndT({2, 3}),    ndT({3, 3})));
        ASSERT_NO_THROW(mm(ndT({1, 2, 3}), ndT({3, 1, 3})));
        ASSERT_NO_THROW(mm(ndT({3,}),      ndT({3, 2})));

        // Wrong shapes
        ASSERT_THROW(mm(ndT({2,}),   ndT({1,})), std::runtime_error);
        ASSERT_THROW(mm(ndT({1,}),   ndT({2,})), std::runtime_error);
        ASSERT_THROW(mm(ndT({3, 3}), ndT({2, 4})), std::runtime_error);

        // Valid output
        {   // (1,) x (1,) -> (1,)
            auto A = r_<int>({5,}).template cast<TypeParam>();
            auto B = r_<int>({3,}).template cast<TypeParam>();
            auto C_gt = r_<int>({15,}).template cast<TypeParam>();
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({1,}));
            ASSERT_TRUE(allclose(C, C_gt));}
        {   // (3,) x (3,) -> (1,)
            auto A = r_<int>({5, 3, 4}).template cast<TypeParam>();
            auto B = r_<int>({3, 1, 2}).template cast<TypeParam>();
            auto C_gt = r_<int>({26,}).template cast<TypeParam>();
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({1,}));
            ASSERT_TRUE(allclose(C, C_gt));}
        {   // (2, 3) x (3,) -> (2,)
            auto A = (r_<int>({1, 2, 3,
                               4, 5, 6})
                      .reshape({2, 3})
                      .template cast<TypeParam>());
            auto B = r_<int>({5, 7, 2}).template cast<TypeParam>();
            auto C_gt = r_<int>({25, 67}).template cast<TypeParam>();
            auto C = mm(A, B);
            ASSERT_EQ(C.shape(), shape_t({2,}));
            ASSERT_TRUE(allclose(C, C_gt));}
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
            ASSERT_TRUE(allclose(C, C_gt));}
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
            ASSERT_TRUE(allclose(C, C_gt));}
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
            ASSERT_TRUE(allclose(C, C_gt));}
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
            ASSERT_TRUE(allclose(C, C_gt));}

        // Valid output (buffer provided)
        {   // (1,) x (1,) -> (1,)
            auto A = r_<int>({5,}).template cast<TypeParam>();
            auto B = r_<int>({3,}).template cast<TypeParam>();
            auto C_gt = r_<int>({15,}).template cast<TypeParam>();
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = mm(A, B, &C);
            ASSERT_TRUE(Z.equals(C));
            ASSERT_EQ(C.shape(), shape_t({1,}));
            ASSERT_TRUE(allclose(C, C_gt));}
        {   // (3,) x (3,) -> (1,)
            auto A = r_<int>({5, 3, 4}).template cast<TypeParam>();
            auto B = r_<int>({3, 1, 2}).template cast<TypeParam>();
            auto C_gt = r_<int>({26,}).template cast<TypeParam>();
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = mm(A, B, &C);
            ASSERT_TRUE(Z.equals(C));
            ASSERT_EQ(C.shape(), shape_t({1,}));
            ASSERT_TRUE(allclose(C, C_gt));}
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
            ASSERT_TRUE(allclose(C, C_gt));}
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
            ASSERT_TRUE(allclose(C, C_gt));}
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
            ASSERT_TRUE(allclose(C, C_gt));}
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
            ASSERT_TRUE(allclose(C, C_gt));}
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
            ASSERT_TRUE(allclose(C, C_gt));}
    }

    TYPED_TEST_P(TestNdLinalgIntFloatComplex, TestBMM) {
        using ndT = ndarray<TypeParam>;

        // Wrong shapes
        ASSERT_THROW(bmm(ndT({2, 3}),       ndT({4, 5})), std::runtime_error);
        ASSERT_THROW(bmm(ndT({1, 2, 3, 4}), ndT({3,})), std::runtime_error);

        { // Non-contiguous buffers
            auto C = ndarray<TypeParam>({4, 10})({util::slice(0, 4, 2),
                                                  util::slice(0, 10, 2)});
            ASSERT_THROW(bmm(ndT({2, 3}), ndT({3, 5}), &C), std::runtime_error);
        }

        // Valid Output
        {   // (2, 3) x (3, 4) -> (1, 2, 4)
            auto A = (r_<int>({7, 8, 4,
                               3, 8, 7})
                      .reshape({2, 3})
                      .template cast<TypeParam>());
            auto B = (r_<int>({7, 7, 4, 5,
                               8, 5, 0, 1,
                               3, 7, 0, 7})
                      .reshape({3, 4})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({125, 117,  28,  71,
                                  106, 110,  12,  72})
                         .reshape({1, 2, 4})
                         .template cast<TypeParam>());
            auto C = bmm(A, B);
            ASSERT_EQ(C.shape(), C_gt.shape());
            ASSERT_TRUE(allclose(C, C_gt));}
        {   // (3, 1, 4) x (1, 4, 1) -> (3, 1, 1)
            auto A = (r_<int>({2, 1, 1, 0,
                               4, 1, 0, 4,
                               2, 9, 0, 2})
                      .reshape({3, 1, 4})
                      .template cast<TypeParam>());
            auto B = (r_<int>({5, 3, 2, 2})
                      .reshape({1, 4, 1})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({15, 31, 41})
                         .reshape({3, 1, 1})
                         .template cast<TypeParam>());
            auto C = bmm(A, B);
            ASSERT_EQ(C.shape(), C_gt.shape());
            ASSERT_TRUE(allclose(C, C_gt));}
        {   // (1, 2, 3) x (3, 3, 2) -> (3, 2, 2)
            auto A = (r_<int>({2, 0, 5,
                               0, 4, 0})
                      .reshape({1, 2, 3})
                      .template cast<TypeParam>());
            auto B = (r_<int>({3, 6,
                               7, 8,
                               5, 1,

                               7, 1,
                               8, 5,
                               4, 4,

                               1, 6,
                               6, 6,
                               6, 2})
                      .reshape({3, 3, 2})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({31, 17,
                                  28, 32,

                                  34, 22,
                                  32, 20,

                                  32, 22,
                                  24, 24})
                         .reshape({3, 2, 2})
                         .template cast<TypeParam>());
            auto C = bmm(A, B);
            ASSERT_EQ(C.shape(), C_gt.shape());
            ASSERT_TRUE(allclose(C, C_gt));}

        // Valid Output (buffer)
        {   // (2, 3) x (3, 4) -> (1, 2, 4)
            auto A = (r_<int>({7, 8, 4,
                               3, 8, 7})
                      .reshape({2, 3})
                      .template cast<TypeParam>());
            auto B = (r_<int>({7, 7, 4, 5,
                               8, 5, 0, 1,
                               3, 7, 0, 7})
                      .reshape({3, 4})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({125, 117,  28,  71,
                                  106, 110,  12,  72})
                         .reshape({1, 2, 4})
                         .template cast<TypeParam>());
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = bmm(A, B, &C);
            ASSERT_TRUE(C.equals(Z));

            ASSERT_EQ(C.shape(), C_gt.shape());
            ASSERT_TRUE(allclose(C, C_gt));}
        {   // (3, 1, 4) x (1, 4, 1) -> (3, 1, 1)
            auto A = (r_<int>({2, 1, 1, 0,
                               4, 1, 0, 4,
                               2, 9, 0, 2})
                      .reshape({3, 1, 4})
                      .template cast<TypeParam>());
            auto B = (r_<int>({5, 3, 2, 2})
                      .reshape({1, 4, 1})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({15, 31, 41})
                         .reshape({3, 1, 1})
                         .template cast<TypeParam>());
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = bmm(A, B, &C);
            ASSERT_TRUE(C.equals(Z));

            ASSERT_EQ(C.shape(), C_gt.shape());
            ASSERT_TRUE(allclose(C, C_gt));}
        {   // (1, 2, 3) x (3, 3, 2) -> (3, 2, 2)
            auto A = (r_<int>({2, 0, 5,
                               0, 4, 0})
                      .reshape({1, 2, 3})
                      .template cast<TypeParam>());
            auto B = (r_<int>({3, 6,
                               7, 8,
                               5, 1,

                               7, 1,
                               8, 5,
                               4, 4,

                               1, 6,
                               6, 6,
                               6, 2})
                      .reshape({3, 3, 2})
                      .template cast<TypeParam>());
            auto C_gt = (r_<int>({31, 17,
                                  28, 32,

                                  34, 22,
                                  32, 20,

                                  32, 22,
                                  24, 24})
                         .reshape({3, 2, 2})
                         .template cast<TypeParam>());
            ndarray<TypeParam> C(C_gt.shape());
            auto Z = bmm(A, B, &C);
            ASSERT_TRUE(C.equals(Z));

            ASSERT_EQ(C.shape(), C_gt.shape());
            ASSERT_TRUE(allclose(C, C_gt));}
    }

    // Initialize tests ====================================================
    REGISTER_TYPED_TEST_CASE_P(TestNdLinalgIntFloatComplex,
                               TestMM,
                               TestBMM);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdLinalgIntFloatComplex, MyIntFloatComplexTypes);
}

#endif // TEST_NDLINALG_CPP
