// ############################################################################
// test_ndutil.cpp
// ===============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDUTIL_CPP
#define TEST_NDUTIL_CPP

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
}}

#endif // TEST_NDUTIL_CPP
