// ############################################################################
// test_ndutil.cpp
// ===============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDUTIL_CPP
#define TEST_NDUTIL_CPP

#include <complex>
#include <functional>
#include <limits>

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd { namespace util {
    TEST(TestNdUtil, TestPredictShape) {
        shape_t const A1({2, 3});
        shape_t const B1({3});
        shape_t const C1 = predict_shape(A1, B1);
        ASSERT_EQ(C1, shape_t({2, 3}));

        shape_t const A2({2, 1});
        shape_t const B2({3});
        shape_t const C2 = predict_shape(A2, B2);
        ASSERT_EQ(C2, shape_t({2, 3}));

        shape_t const A3({2, 3, 5, 7});
        shape_t const B3({2, 3, 5, 7});
        shape_t const C3 = predict_shape(A3, B3);
        ASSERT_EQ(C3, shape_t({2, 3, 5, 7}));
    }

    TEST(TestNdUtil, TestSliceConstructor) {
        slice a;
        ASSERT_EQ(a.start(), 0);
        ASSERT_EQ(a.stop(), std::numeric_limits<int>::max());
        ASSERT_EQ(a.step(), 1);

        slice b(5);
        ASSERT_EQ(b.start(), 0);
        ASSERT_EQ(b.stop(), 5);
        ASSERT_EQ(b.step(), 1);

        slice c(5, 6);
        ASSERT_EQ(c.start(), 5);
        ASSERT_EQ(c.stop(), 6);
        ASSERT_EQ(c.step(), 1);

        slice d(5, 12, 3);
        ASSERT_EQ(d.start(), 5);
        ASSERT_EQ(d.stop(), 12);
        ASSERT_EQ(d.step(), 3);

        slice e(5, 6, -1);
        ASSERT_EQ(e.start(), 5);
        ASSERT_EQ(e.stop(), 6);
        ASSERT_EQ(e.step(), -1);
    }

    TEST(TestNdUtil, TestMapLimits) {
        // Entire range
        int const l_1 = 12;
        slice in_1;
        slice const out_1 = in_1.map_limits(l_1);
        ASSERT_EQ(out_1.start(), 0);
        ASSERT_EQ(out_1.stop(), l_1);
        ASSERT_EQ(out_1.step(), 1);

        // stop > 0
        int const l_2 = 12;
        int const stop_2 = 4;
        slice in_2(stop_2);
        slice const out_2 = in_2.map_limits(l_2);
        ASSERT_EQ(out_2.start(), 0);
        ASSERT_EQ(out_2.stop(), stop_2);
        ASSERT_EQ(out_2.step(), 1);

        // stop == 0
        int const l_3 = 12;
        int const stop_3 = 0;
        slice in_3(stop_3);
        slice const out_3 = in_3.map_limits(l_3);
        ASSERT_EQ(out_3.start(), 0);
        ASSERT_EQ(out_3.stop(), 0);
        ASSERT_EQ(out_3.step(), 1);

        // start < stop
        int const l_4 = 12;
        int const start_4 = 2;
        int const stop_4 = 4;
        slice in_4(start_4, stop_4);
        slice const out_4 = in_4.map_limits(l_4);
        ASSERT_EQ(out_4.start(), start_4);
        ASSERT_EQ(out_4.stop(), stop_4);
        ASSERT_EQ(out_4.step(), 1);

        // start == stop
        int const l_5 = 12;
        int const start_5 = 2;
        int const stop_5 = 2;
        slice in_5(start_5, stop_5);
        slice const out_5 = in_5.map_limits(l_5);
        ASSERT_EQ(out_5.start(), start_5);
        ASSERT_EQ(out_5.stop(), stop_5);
        ASSERT_EQ(out_5.step(), 1);

        // start > stop
        int const l_6 = 12;
        int const start_6 = 3;
        int const stop_6 = 2;
        slice in_6(start_6, stop_6);
        slice const out_6 = in_6.map_limits(l_6);
        ASSERT_EQ(out_6.start(), 0);
        ASSERT_EQ(out_6.stop(), 0);
        ASSERT_EQ(out_6.step(), 1);

        // start < stop, step > 0
        int const l_7 = 12;
        int const start_7 = 2;
        int const stop_7 = 5;
        int const step_7 = 2;
        slice in_7(start_7, stop_7, step_7);
        slice const out_7 = in_7.map_limits(l_7);
        ASSERT_EQ(out_7.start(), start_7);
        ASSERT_EQ(out_7.stop(), stop_7);
        ASSERT_EQ(out_7.step(), step_7);

        // start == stop, step > 0
        int const l_8 = 12;
        int const start_8 = 2;
        int const stop_8 = 2;
        int const step_8 = 1;
        slice in_8(start_8, stop_8, step_8);
        slice const out_8 = in_8.map_limits(l_8);
        ASSERT_EQ(out_8.start(), start_8);
        ASSERT_EQ(out_8.stop(), stop_8);
        ASSERT_EQ(out_8.step(), step_8);

        // start > stop, step > 0
        int const l_9 = 12;
        int const start_9 = 4;
        int const stop_9 = 2;
        int const step_9 = 1;
        slice in_9(start_9, stop_9, step_9);
        slice const out_9 = in_9.map_limits(l_9);
        ASSERT_EQ(out_9.start(), 0);
        ASSERT_EQ(out_9.stop(), 0);
        ASSERT_EQ(out_9.step(), step_9);

        // start > stop, step < 0
        int const l_10 = 12;
        int const start_10 = 4;
        int const stop_10 = 2;
        int const step_10 = -1;
        slice in_10(start_10, stop_10, step_10);
        slice const out_10 = in_10.map_limits(l_10);
        ASSERT_EQ(out_10.start(), start_10);
        ASSERT_EQ(out_10.stop(), stop_10);
        ASSERT_EQ(out_10.step(), step_10);

        // start == stop, step < 0
        int const l_11 = 12;
        int const start_11 = 4;
        int const stop_11 = 4;
        int const step_11 = -1;
        slice in_11(start_11, stop_11, step_11);
        slice const out_11 = in_11.map_limits(l_11);
        ASSERT_EQ(out_11.start(), start_11);
        ASSERT_EQ(out_11.stop(), stop_11);
        ASSERT_EQ(out_11.step(), step_11);

        // start < stop, step < 0
        int const l_12 = 12;
        int const start_12 = 4;
        int const stop_12 = 8;
        int const step_12 = -1;
        slice in_12(start_12, stop_12, step_12);
        slice const out_12 = in_12.map_limits(l_12);
        ASSERT_EQ(out_12.start(), 0);
        ASSERT_EQ(out_12.stop(), 0);
        ASSERT_EQ(out_12.step(), step_12);

        int const l_13 = 12;
        int const start_13 = 20;
        int const stop_13 = -3;
        int const step_13 = -1;
        slice in_13(start_13, stop_13, step_13);
        slice const out_13 = in_13.map_limits(l_13);
        ASSERT_EQ(out_13.start(), 11);
        ASSERT_EQ(out_13.stop(), -1);
        ASSERT_EQ(out_13.step(), -1);
    }

    template <typename T>
    class TestNdUtilApply : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdUtilApply);

    TYPED_TEST_P(TestNdUtilApply, TestApplyUnary) {
        ndarray<TypeParam> in  = zeros<TypeParam>({5, 3, 4});
        ndarray<TypeParam> out = zeros<TypeParam>({5, 3, 4});

        auto ufunc = [](TypeParam const& _x) -> TypeParam { return _x + static_cast<TypeParam>(1.0); };
        apply(ufunc, &in, &out);

        for(auto it_in = in.begin(), it_out = out.begin();
            it_in != in.end();
            ++it_in, ++it_out) {
            TypeParam const tested_elem = *it_out;
            TypeParam const correct_elem = (*it_in) + static_cast<TypeParam>(1.0);
            ASSERT_EQ(correct_elem, tested_elem);
        }
    }

    TYPED_TEST_P(TestNdUtilApply, TestApplyBinary) {
       // No broadcasting case
       {ndarray<TypeParam> in_1  = zeros<TypeParam>({5, 3, 4});
        ndarray<TypeParam> in_2  = ones<TypeParam>({5, 3, 4});
        ndarray<TypeParam> out = zeros<TypeParam>({5, 3, 4});

        apply(std::plus<TypeParam>(), &in_1, &in_2, &out);

        for(auto it_in_1 = in_1.begin(), it_in_2 = in_2.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_in_1, ++it_in_2, ++it_out) {
            TypeParam const tested_elem = *it_out;
            TypeParam const correct_elem = (*it_in_1) + (*it_in_2);
            ASSERT_EQ(correct_elem, tested_elem);
        }}

       // Broadcasting case
       {ndarray<TypeParam> in_1  = zeros<TypeParam>({5, 1, 4});
        ndarray<TypeParam> in_2  = ones<TypeParam>({3, 1});
        ndarray<TypeParam> out = zeros<TypeParam>({5, 3, 4});

        apply(std::plus<TypeParam>(), &in_1, &in_2, &out);

        ndarray<TypeParam> in_1_bcast = in_1.broadcast_to({5, 3, 4});
        ndarray<TypeParam> in_2_bcast = in_2.broadcast_to({5, 3, 4});
        for(auto it_in_1 = in_1_bcast.begin(), it_in_2 = in_2_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_in_1, ++it_in_2, ++it_out) {
            TypeParam const tested_elem = *it_out;
            TypeParam const correct_elem = (*it_in_1) + (*it_in_2);
            ASSERT_EQ(correct_elem, tested_elem);
        }}
    }

    typedef ::testing::Types<int, size_t,
                             float, double,
                             std::complex<float>, std::complex<double>> MyNdUtilApplyTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdUtilApply,
                               TestApplyUnary,
                               TestApplyBinary);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdUtilApply, MyNdUtilApplyTypes);

    template <typename T>
    class TestNdUtilReduce3D : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdUtilReduce3D);
    TYPED_TEST_P(TestNdUtilReduce3D, TestReduce3D) {
        // axis = 0
       {auto x = nd::arange<TypeParam>(0, 3 * 4 * 5, 1).reshape({3, 4, 5});
        auto y = nd::zeros<TypeParam>({1, 4, 5});

        reduce3D(std::plus<TypeParam>(), &x, &y, 0, static_cast<TypeParam>(0));

        TypeParam correct_elem = 60;
        for(size_t i = 0; i < y.shape()[1]; ++i, correct_elem += 0) {
            for(size_t j = 0; j < y.shape()[2]; ++j, correct_elem += 3) {
                TypeParam const& tested_elem = y[{0, i, j}];
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }}

        // axis = 1
       {auto x = nd::arange<TypeParam>(0, 3 * 4 * 5, 1).reshape({3, 4, 5});
        auto y = nd::zeros<TypeParam>({3, 1, 5});

        reduce3D(std::plus<TypeParam>(), &x, &y, 1, static_cast<TypeParam>(0));

        TypeParam correct_elem = 30;
        for(size_t i = 0; i < y.shape()[0]; ++i, correct_elem += 60) {
            for(size_t j = 0; j < y.shape()[2]; ++j, correct_elem += 4) {
                TypeParam const& tested_elem = y[{i, 0, j}];
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }}

        // axis = 2
       {auto x = nd::arange<TypeParam>(0, 3 * 4 * 5, 1).reshape({3, 4, 5});
        auto y = nd::zeros<TypeParam>({3, 4, 1});

        reduce3D(std::plus<TypeParam>(), &x, &y, 2, static_cast<TypeParam>(0));

        TypeParam correct_elem = 10;
        for(size_t i = 0; i < y.shape()[0]; ++i, correct_elem += 0) {
            for(size_t j = 0; j < y.shape()[1]; ++j, correct_elem += 25) {
                TypeParam const& tested_elem = y[{i, j, 0}];
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }}
    }

    typedef ::testing::Types<int, float, double> MyNdUtilReduce3DTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdUtilReduce3D,
                               TestReduce3D);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdUtilReduce3D, MyNdUtilReduce3DTypes);
}}

#endif // TEST_NDUTIL_CPP
