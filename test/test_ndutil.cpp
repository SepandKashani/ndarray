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
#include <sstream>
#include <vector>

#include <gtest/gtest.h>
#include "test_type.hpp"

#include "ndarray/ndarray.hpp"

namespace nd::util {
    TEST(TestNdUtil, TestPredictShapeBroadcast) {
        shape_t const A1({2, 3});
        shape_t const B1({3});
        shape_t const C1 = predict_shape_broadcast(A1, B1);
        ASSERT_EQ(C1, shape_t({2, 3}));

        shape_t const A2({2, 1});
        shape_t const B2({3});
        shape_t const C2 = predict_shape_broadcast(A2, B2);
        ASSERT_EQ(C2, shape_t({2, 3}));

        shape_t const A3({2, 3, 5, 7});
        shape_t const B3({2, 3, 5, 7});
        shape_t const C3 = predict_shape_broadcast(A3, B3);
        ASSERT_EQ(C3, shape_t({2, 3, 5, 7}));
    }

    TEST(TestNdUtil, TestPredictShapeReduction) {
        shape_t const A({3, 5});
        ASSERT_EQ(predict_shape_reduction(A, 0), shape_t({1, 5}));
        ASSERT_EQ(predict_shape_reduction(A, 1), shape_t({3, 1}));

        shape_t const B({5});
        ASSERT_EQ(predict_shape_reduction(B, 0), shape_t({1}));
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

    REGISTER_TYPED_TEST_CASE_P(TestNdUtilApply,
                               TestApplyUnary,
                               TestApplyBinary);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdUtilApply, MyIntFloatComplexTypes);

    template <typename T>
    class TestNdUtilReduce : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdUtilReduce);
    TYPED_TEST_P(TestNdUtilReduce, TestReduce3D) {
        // axis = 0
       {auto x = (arange<TypeParam>(0, 3 * 4 * 5, 1)
                  .reshape(shape_t({3, 4, 5})));
        auto y = zeros<TypeParam>(shape_t({1, 4, 5}));

        reduce3D(std::plus<TypeParam>(), &x, &y, 0, static_cast<TypeParam>(0));

        TypeParam correct_elem = 60;
        for(size_t i = 0; i < y.shape()[1]; ++i, correct_elem += 0) {
            for(size_t j = 0; j < y.shape()[2]; ++j, correct_elem += 3) {
                TypeParam const& tested_elem = y[{0, i, j}];
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }}

        // axis = 1
       {auto x = (arange<TypeParam>(0, 3 * 4 * 5, 1)
                  .reshape(shape_t({3, 4, 5})));
        auto y = zeros<TypeParam>({3, 1, 5});

        reduce3D(std::plus<TypeParam>(), &x, &y, 1, static_cast<TypeParam>(0));

        TypeParam correct_elem = 30;
        for(size_t i = 0; i < y.shape()[0]; ++i, correct_elem += 60) {
            for(size_t j = 0; j < y.shape()[2]; ++j, correct_elem += 4) {
                TypeParam const& tested_elem = y[{i, 0, j}];
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }}

        // axis = 2
       {auto x = (arange<TypeParam>(0, 3 * 4 * 5, 1)
                  .reshape(shape_t({3, 4, 5})));
        auto y = zeros<TypeParam>({3, 4, 1});

        reduce3D(std::plus<TypeParam>(), &x, &y, 2, static_cast<TypeParam>(0));

        TypeParam correct_elem = 10;
        for(size_t i = 0; i < y.shape()[0]; ++i, correct_elem += 0) {
            for(size_t j = 0; j < y.shape()[1]; ++j, correct_elem += 25) {
                TypeParam const& tested_elem = y[{i, j, 0}];
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }}
    }

    TYPED_TEST_P(TestNdUtilReduce, TestReduce) {
        shape_t const shape({2, 2, 2, 2});
        auto const x = arange<TypeParam>(0, 16, 1).reshape(shape);

       {ndarray<TypeParam> y_axis0(shape_t({1, 2, 2, 2}));
        reduce(std::plus<TypeParam>(),
               const_cast<ndarray<TypeParam>*>(&x),
               &y_axis0,
               0,
               static_cast<TypeParam>(0));
        auto y_correct_axis0 = (r_<TypeParam>({8, 10, 12, 14, 16, 18, 20, 22})
                                .reshape(shape_t({1, 2, 2, 2})));
        ASSERT_EQ(y_axis0.shape(), y_correct_axis0.shape());
        bool closeness = allclose(y_axis0.template cast<float>(),
                                  y_correct_axis0.template cast<float>());
        ASSERT_TRUE(closeness);}

       {ndarray<TypeParam> y_axis1(shape_t({2, 1, 2, 2}));
        reduce(std::plus<TypeParam>(),
               const_cast<ndarray<TypeParam>*>(&x),
               &y_axis1,
               1,
               static_cast<TypeParam>(0));
        auto y_correct_axis1 = (r_<TypeParam>({4, 6, 8, 10, 20, 22, 24, 26})
                                .reshape(shape_t({2, 1, 2, 2})));
        ASSERT_EQ(y_axis1.shape(), y_correct_axis1.shape());
        bool closeness = allclose(y_axis1.template cast<float>(),
                                  y_correct_axis1.template cast<float>());
        ASSERT_TRUE(closeness);}

       {ndarray<TypeParam> y_axis2(shape_t({2, 2, 1, 2}));
        reduce(std::plus<TypeParam>(),
               const_cast<ndarray<TypeParam>*>(&x),
               &y_axis2,
               2,
               static_cast<TypeParam>(0));
        auto y_correct_axis2 = (r_<TypeParam>({2, 4, 10, 12, 18, 20, 26, 28})
                                .reshape(shape_t({2, 2, 1, 2})));
        ASSERT_EQ(y_axis2.shape(), y_correct_axis2.shape());
        bool closeness = allclose(y_axis2.template cast<float>(),
                                  y_correct_axis2.template cast<float>());
        ASSERT_TRUE(closeness);}

       {ndarray<TypeParam> y_axis3(shape_t({2, 2, 2, 1}));
        reduce(std::plus<TypeParam>(),
               const_cast<ndarray<TypeParam>*>(&x),
               &y_axis3,
               3,
               static_cast<TypeParam>(0));
        auto y_correct_axis3 = (r_<TypeParam>({1, 5, 9, 13, 17, 21, 25, 29})
                                .reshape(shape_t({2, 2, 2, 1})));
        ASSERT_EQ(y_axis3.shape(), y_correct_axis3.shape());
        bool closeness = allclose(y_axis3.template cast<float>(),
                                  y_correct_axis3.template cast<float>());
        ASSERT_TRUE(closeness);}
    }

    TYPED_TEST_P(TestNdUtilReduce, TestReduce1D) {
        auto const x = arange<TypeParam>(0, 10, 1);

       {ndarray<TypeParam> y_axis0(shape_t({1}));
        reduce(std::plus<TypeParam>(),
               const_cast<ndarray<TypeParam>*>(&x),
               &y_axis0,
               0,
               static_cast<TypeParam>(0));
        auto y_correct_axis0 = r_<TypeParam>({45});
        ASSERT_EQ(y_axis0.shape(), y_correct_axis0.shape());
        bool closeness = allclose(y_axis0.template cast<float>(),
                                  y_correct_axis0.template cast<float>());
        ASSERT_TRUE(closeness);}
    }
    typedef ::testing::Types<int, float, double> MyNdUtilReduceTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdUtilReduce,
                               TestReduce3D,
                               TestReduce,
                               TestReduce1D);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdUtilReduce, MySignedIntFloatTypes);
}

namespace nd::util::interop {
    template <typename T>
    class TestNdUtilInterop : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdUtilInterop);

    TYPED_TEST_P(TestNdUtilInterop, TestIsEigenMappable) {
        // 1d
       {auto x = (arange<int>(0, 5 * 3, 1)
                  .template cast<TypeParam>());
        ASSERT_TRUE(is_eigen_mappable(&x));}

        // 2d
       {auto x = (arange<int>(0, 5 * 3, 1)
                  .reshape(shape_t({5, 3}))
                  .template cast<TypeParam>());
        ASSERT_TRUE(is_eigen_mappable(&x));}

        // 3d
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape(shape_t({5, 3, 4}))
                  .template cast<TypeParam>());
        ASSERT_FALSE(is_eigen_mappable(&x));}

        // Strided
       {auto x = (arange<int>(0, 10 * 10, 1)
                  .reshape(shape_t({10, 10}))
                  .template cast<TypeParam>());
        auto y = x({slice(1, 6, 2), slice(2, 10, 3)});
        ASSERT_TRUE(is_eigen_mappable(&y));}

        // Strided, overlapping
       {auto x = (arange<int>(0, 100, 1)
                  .template cast<TypeParam>());
        auto y = ndarray<TypeParam>(reinterpret_cast<byte_t*>(x.data()),
                                    shape_t({200}),
                                    stride_t({sizeof(TypeParam)/2}));
        ASSERT_FALSE(is_eigen_mappable(&y));}

        // Negative strides
       {auto x = (arange<int>(0, 10 * 10, 1)
                  .reshape(shape_t({10, 10}))
                  .template cast<TypeParam>());
        auto y = x({slice(10, -1, -1), slice(2, 10, 3)});

        ASSERT_TRUE(y.size() > 0);
        ASSERT_FALSE(is_eigen_mappable(&y));}
    }

    TYPED_TEST_P(TestNdUtilInterop, TestAsEigenArray) {
        // 1d
       {size_t const N = 5;
        auto x = (arange<int>(0, N, 1)
                  .template cast<TypeParam>());
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);

        ASSERT_EQ(size_t(y->rows()), size_t(1));
        ASSERT_EQ(size_t(y->cols()), size_t(N));

        for(size_t i = 0; i < N; ++i) {
            TypeParam const& gt = x[{i}];
            TypeParam const& res = (*y)(0, i);
            ASSERT_EQ(gt, res);
        }}

        // 2d
       {size_t const N(5), M(3);
        auto x = (arange<int>(0, N * M, 1)
                  .reshape(shape_t({N, M}))
                  .template cast<TypeParam>());
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);

        ASSERT_EQ(size_t(y->rows()), N);
        ASSERT_EQ(size_t(y->cols()), M);

        for(size_t i = 0; i < N; ++i) {
            for(size_t j = 0; j < M; ++j) {
                TypeParam const& gt = x[{i, j}];
                TypeParam const& res = (*y)(i, j);
                ASSERT_EQ(gt, res);
            }
        }}

        // 2d
       {size_t const N(5), M(3);
        auto x = (arange<int>(0, N * M, 1)
                  .reshape(shape_t({N, M}))
                  .template cast<TypeParam>());
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);

        ASSERT_EQ(size_t(y->rows()), N);
        ASSERT_EQ(size_t(y->cols()), M);

        for(size_t i = 0; i < N; ++i) {
            for(size_t j = 0; j < M; ++j) {
                TypeParam const& gt = x[{i, j}];
                TypeParam const& res = (*y)(i, j);
                ASSERT_EQ(gt, res);
            }
        }}

        // Strided
       {size_t const N(10), M(10);
        auto x = (arange<int>(0, N * M, 1)
                  .reshape(shape_t({N, M}))
                  .template cast<TypeParam>()
                  .operator()({slice(1, 6, 2), slice(2, 10, 3)}));
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);

        ASSERT_EQ(size_t(y->rows()), size_t(x.shape()[0]));
        ASSERT_EQ(size_t(y->cols()), size_t(x.shape()[1]));

        for(size_t i = 0; i < x.shape()[0]; ++i) {
            for(size_t j = 0; j < x.shape()[1]; ++j) {
                TypeParam const& gt = x[{i, j}];
                TypeParam const& res = (*y)(i, j);
                ASSERT_EQ(gt, res);
            }
        }}
    }

    TYPED_TEST_P(TestNdUtilInterop, TestAsNdarray) {
        // 1d: output ndarray is independent of input ndarray.
       {size_t const N = 5;
        auto x = (arange<int>(0, N, 1)
                  .template cast<TypeParam>());
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);
        auto z = asndarray(*y);

        ASSERT_FALSE(z.equals(x));
        ASSERT_EQ(z.base().use_count(), 1);
        ASSERT_EQ(z.shape(), shape_t({1, N}));

        for(auto it_x = x.begin(), it_z = z.begin(); it_x != x.end(); ++it_x, ++it_z) {
            TypeParam const& gt = (*it_x);
            TypeParam const& res = (*it_z);
            ASSERT_EQ(gt, res);
        }}

        // 2d: output ndarray is independent of input ndarray.
       {size_t const N(5), M(3);
        auto x = (arange<int>(0, N * M, 1)
                  .reshape(shape_t({N, M}))
                  .template cast<TypeParam>());
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);
        auto z = asndarray(*y);

        ASSERT_FALSE(z.equals(x));
        ASSERT_EQ(z.base().use_count(), 1);
        ASSERT_EQ(z.shape(), shape_t({N, M}));

        for(auto it_x = x.begin(), it_z = z.begin(); it_x != x.end(); ++it_x, ++it_z) {
            TypeParam const& gt = (*it_x);
            TypeParam const& res = (*it_z);
            ASSERT_EQ(gt, res);
        }}

        // Transforms possible with expressions.
       {size_t const N(5), M(3);
        auto x = (arange<int>(0, N * M, 1)
                  .reshape(shape_t({N, M}))
                  .template cast<TypeParam>());
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);
        auto z = asndarray(((*y) + TypeParam(2.0)).template cast<TypeParam>());

        ASSERT_FALSE(z.equals(x));
        ASSERT_EQ(z.base().use_count(), 1);
        ASSERT_EQ(z.shape(), shape_t({N, M}));

        for(auto it_x = x.begin(), it_z = z.begin(); it_x != x.end(); ++it_x, ++it_z) {
            TypeParam const& gt = (*it_x) + TypeParam(2.0);
            TypeParam const& res = (*it_z);
            ASSERT_EQ(gt, res);
        }}
    }

    REGISTER_TYPED_TEST_CASE_P(TestNdUtilInterop,
                               TestIsEigenMappable,
                               TestAsEigenArray,
                               TestAsNdarray);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdUtilInterop, MyArithmeticTypes);

    TEST(TestNdUtil, TestOperatorPrint) {
        // soft tests to make sure printed outputs look ok.
        size_t const N(3), M(2);

       {auto x = (nd::arange<int>(0, N * M, 1)
                  .reshape({N, M})
                  .template cast<int>());
        std::stringstream res;
        res << x;

        ASSERT_EQ(res.str(), "[[0, 1]\n [2, 3]\n [4, 5]]");}

       {auto x = (nd::arange<int>(0, N * M, 1)
                  .reshape({N, M})
                  .template cast<float>()) + nd::ndarray<float>(0.5);
        std::stringstream res;
        res << x;
        ASSERT_EQ(res.str(), "[[0.5, 1.5]\n [2.5, 3.5]\n [4.5, 5.5]]");}

       {auto x = (nd::arange<int>(0, N * M, 1)
                  .template cast<int>());
        std::stringstream res;
        res << x;
        ASSERT_EQ(res.str(), "[0, 1, 2, 3, 4, 5]");}

       {auto x = (nd::arange<int>(0, N * M, 1)
                  .template cast<std::complex<double>>()) + nd::ndarray<std::complex<double>>(0.5);
        std::stringstream res;
        res << x;
        ASSERT_EQ(res.str(), "[(0.5,0), (1.5,0), (2.5,0), (3.5,0), (4.5,0), (5.5,0)]");}
    }
}

#endif // TEST_NDUTIL_CPP
