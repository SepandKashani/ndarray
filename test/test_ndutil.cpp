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
       {shape_t const A({2, 3});
        shape_t const B({3});
        shape_t const C = predict_shape_broadcast(A, B);
        ASSERT_EQ(C, shape_t({2, 3}));}

       {shape_t const A({2, 1});
        shape_t const B({3});
        shape_t const C = predict_shape_broadcast(A, B);
        ASSERT_EQ(C, shape_t({2, 3}));}

       {shape_t const A({2, 3, 5, 7});
        shape_t const B({2, 3, 5, 7});
        shape_t const C = predict_shape_broadcast(A, B);
        ASSERT_EQ(C, shape_t({2, 3, 5, 7}));}
    }

    TEST(TestNdUtil, TestPredictShapeReduction) {
       {shape_t const A({3, 5});
        ASSERT_EQ(predict_shape_reduction(A, 0), shape_t({1, 5}));
        ASSERT_EQ(predict_shape_reduction(A, 1), shape_t({3, 1}));}

       {shape_t const A({5});
        ASSERT_EQ(predict_shape_reduction(A, 0), shape_t({1}));}
    }

    TEST(TestNdUtil, TestSliceConstructor) {
       {slice s;
        ASSERT_EQ(s.start(), 0);
        ASSERT_EQ(s.stop(), std::numeric_limits<int>::max());
        ASSERT_EQ(s.step(), 1);}

       {slice s(5);
        ASSERT_EQ(s.start(), 0);
        ASSERT_EQ(s.stop(), 5);
        ASSERT_EQ(s.step(), 1);}

       {slice s(5, 6);
        ASSERT_EQ(s.start(), 5);
        ASSERT_EQ(s.stop(), 6);
        ASSERT_EQ(s.step(), 1);}

       {slice s(5, 12, 3);
        ASSERT_EQ(s.start(), 5);
        ASSERT_EQ(s.stop(), 12);
        ASSERT_EQ(s.step(), 3);}

       {slice s(5, 6, -1);
        ASSERT_EQ(s.start(), 5);
        ASSERT_EQ(s.stop(), 6);
        ASSERT_EQ(s.step(), -1);}
    }

    TEST(TestNdUtil, TestMapLimits) {
        // Entire range
       {int const l = 12;
        slice in;
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), 0);
        ASSERT_EQ(out.stop(), l);
        ASSERT_EQ(out.step(), 1);}

        // stop > 0
       {int const l = 12;
        int const stop = 4;
        slice in(stop);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), 0);
        ASSERT_EQ(out.stop(), stop);
        ASSERT_EQ(out.step(), 1);}

        // stop == 0
       {int const l = 12;
        int const stop = 0;
        slice in(stop);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), 0);
        ASSERT_EQ(out.stop(), 0);
        ASSERT_EQ(out.step(), 1);}

        // start < stop
       {int const l = 12;
        int const start = 2;
        int const stop = 4;
        slice in(start, stop);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), start);
        ASSERT_EQ(out.stop(), stop);
        ASSERT_EQ(out.step(), 1);}

        // start == stop
       {int const l = 12;
        int const start = 2;
        int const stop = 2;
        slice in(start, stop);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), start);
        ASSERT_EQ(out.stop(), stop);
        ASSERT_EQ(out.step(), 1);}

        // start > stop
       {int const l = 12;
        int const start = 3;
        int const stop = 2;
        slice in(start, stop);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), 0);
        ASSERT_EQ(out.stop(), 0);
        ASSERT_EQ(out.step(), 1);}

        // start < stop, step > 0
       {int const l = 12;
        int const start = 2;
        int const stop = 5;
        int const step = 2;
        slice in(start, stop, step);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), start);
        ASSERT_EQ(out.stop(), stop);
        ASSERT_EQ(out.step(), step);}

        // start == stop, step > 0
       {int const l = 12;
        int const start = 2;
        int const stop = 2;
        int const step = 1;
        slice in(start, stop, step);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), start);
        ASSERT_EQ(out.stop(), stop);
        ASSERT_EQ(out.step(), step);}

        // start > stop, step > 0
       {int const l = 12;
        int const start = 4;
        int const stop = 2;
        int const step = 1;
        slice in(start, stop, step);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), 0);
        ASSERT_EQ(out.stop(), 0);
        ASSERT_EQ(out.step(), step);}

        // start > stop, step < 0
       {int const l = 12;
        int const start = 4;
        int const stop = 2;
        int const step = -1;
        slice in(start, stop, step);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), start);
        ASSERT_EQ(out.stop(), stop);
        ASSERT_EQ(out.step(), step);}

        // start == stop, step < 0
       {int const l = 12;
        int const start = 4;
        int const stop = 4;
        int const step = -1;
        slice in(start, stop, step);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), start);
        ASSERT_EQ(out.stop(), stop);
        ASSERT_EQ(out.step(), step);}

        // start < stop, step < 0
       {int const l = 12;
        int const start = 4;
        int const stop = 8;
        int const step = -1;
        slice in(start, stop, step);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), 0);
        ASSERT_EQ(out.stop(), 0);
        ASSERT_EQ(out.step(), step);}

       {int const l = 12;
        int const start = 20;
        int const stop = -3;
        int const step = -1;
        slice in(start, stop, step);
        slice const out = in.map_limits(l);
        ASSERT_EQ(out.start(), 11);
        ASSERT_EQ(out.stop(), -1);
        ASSERT_EQ(out.step(), -1);}
    }

    TEST(TestNdUtil, TestApplyUnary) {
       {ndarray<int> in  = zeros<int>({5, 3, 4});
        ndarray<int> out = zeros<int>({5, 3, 4});
        ndarray<int> gt = ones<int>({5, 3, 4});
        auto ufunc = [](int const& _x) -> int { return _x + 1; };
        apply(ufunc, &in, &out);
        ASSERT_TRUE(allclose(out, gt));}
    }

    TEST(TestNdUtil, TestApplyBinary) {
       // No broadcasting case
       {ndarray<int> in_1  = zeros<int>({5, 3, 4});
        ndarray<int> in_2  = ones<int>({5, 3, 4});
        ndarray<int> out = zeros<int>({5, 3, 4});
        apply(std::plus<int>(), &in_1, &in_2, &out);
        ASSERT_TRUE(allclose(out, in_2));}

       // Broadcasting case
       {ndarray<int> in_1  = zeros<int>({5, 1, 4});
        ndarray<int> in_2  = ones<int>({3, 1});
        ndarray<int> out = zeros<int>({5, 3, 4});
        apply(std::plus<int>(), &in_1, &in_2, &out);
        ASSERT_TRUE(allclose(out, in_1 + in_2));}
    }

    TEST(TestNdUtil, TestReduce3D) {
        // axis = 0
       {auto x = (arange<int>(0, 3 * 4 * 5, 1)
                  .reshape({3, 4, 5}));
        auto y = zeros<int>({1, 4, 5});
        reduce3D(std::plus<int>(), &x, &y, 0, 0);
        auto gt = (r_<int>({ 60,  63,  66,  69,  72,
                            75,  78,  81,  84,  87,
                            90,  93,  96,  99, 102,
                           105, 108, 111, 114, 117})
                   .reshape({1, 4, 5}));
        ASSERT_TRUE(allclose(y, gt));}

        // axis = 1
       {auto x = (arange<int>(0, 3 * 4 * 5, 1)
                  .reshape({3, 4, 5}));
        auto y = zeros<int>({3, 1, 5});
        reduce3D(std::plus<int>(), &x, &y, 1, 0);
        auto gt = (r_<int>({ 30,  34,  38,  42,  46,
                            110, 114, 118, 122, 126,
                            190, 194, 198, 202, 206})
                   .reshape({3, 1, 5}));
        ASSERT_TRUE(allclose(y, gt));}

        // axis = 2
       {auto x = (arange<int>(0, 3 * 4 * 5, 1)
                  .reshape({3, 4, 5}));
        auto y = zeros<int>({3, 4, 1});
        reduce3D(std::plus<int>(), &x, &y, 2, 0);
        auto gt = (r_<int>({ 10,  35,  60,  85,
                            110, 135, 160, 185,
                            210, 235, 260, 285})
                   .reshape({3, 4, 1}));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TEST(TestNdUtil, TestReduce) {
        shape_t const shape({2, 2, 2, 2});
        auto const x = (arange<int>(0, 16, 1)
                        .reshape(shape));

        // axis = 0
       {ndarray<int> y({1, 2, 2, 2});
        reduce(std::plus<int>(), const_cast<ndarray<int>*>(&x), &y, 0, 0);
        auto gt = (r_<int>({8, 10, 12, 14, 16, 18, 20, 22})
                   .reshape({1, 2, 2, 2}));
        ASSERT_TRUE(allclose(y, gt));}

        // axis = 1
       {ndarray<int> y({2, 1, 2, 2});
        reduce(std::plus<int>(), const_cast<ndarray<int>*>(&x), &y, 1, 0);
        auto gt = (r_<int>({4, 6, 8, 10, 20, 22, 24, 26})
                   .reshape({2, 1, 2, 2}));
        ASSERT_TRUE(allclose(y, gt));}

        // axis = 2
       {ndarray<int> y({2, 2, 1, 2});
        reduce(std::plus<int>(), const_cast<ndarray<int>*>(&x), &y, 2, 0);
        auto gt = (r_<int>({2, 4, 10, 12, 18, 20, 26, 28})
                   .reshape({2, 2, 1, 2}));
        ASSERT_TRUE(allclose(y, gt));}

        // axis = 3
       {ndarray<int> y({2, 2, 2, 1});
        reduce(std::plus<int>(), const_cast<ndarray<int>*>(&x), &y, 3, 0);
        auto gt = (r_<int>({1, 5, 9, 13, 17, 21, 25, 29})
                   .reshape({2, 2, 2, 1}));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TEST(TestNdUtil, TestReduce1D) {
        auto const x = arange<int>(0, 10, 1);

       {ndarray<int> y(shape_t({1}));
        reduce(std::plus<int>(), const_cast<ndarray<int>*>(&x), &y, 0, 0);
        auto gt = r_<int>({45});
        ASSERT_TRUE(allclose(y, gt));}
    }
}

namespace nd::util::interop {
    template <typename T>
    class TestNdUtilInteropArithmetic : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdUtilInteropArithmetic);

    TYPED_TEST_P(TestNdUtilInteropArithmetic, TestIsEigenMappable) {
        // 1d
       {auto x = (arange<int>(0, 5 * 3, 1)
                  .template cast<TypeParam>());
        ASSERT_TRUE(is_eigen_mappable(&x));}

        // 2d
       {auto x = (arange<int>(0, 5 * 3, 1)
                  .reshape({5, 3})
                  .template cast<TypeParam>());
        ASSERT_TRUE(is_eigen_mappable(&x));}

        // 3d
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());
        ASSERT_FALSE(is_eigen_mappable(&x));}

        // Strided
       {auto x = (arange<int>(0, 10 * 10, 1)
                  .reshape({10, 10})
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
                  .reshape({10, 10})
                  .template cast<TypeParam>());
        auto y = x({slice(10, -1, -1), slice(2, 10, 3)});

        ASSERT_TRUE(y.size() > 0);
        ASSERT_FALSE(is_eigen_mappable(&y));}
    }

    TYPED_TEST_P(TestNdUtilInteropArithmetic, TestAsEigenArray) {
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
                  .reshape({N, M})
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
                  .reshape({N, M})
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
                  .reshape({N, M})
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

    TYPED_TEST_P(TestNdUtilInteropArithmetic, TestAsNdarray) {
        // 1d: output ndarray is independent of input ndarray.
       {size_t const N = 5;
        auto x = (arange<int>(0, N, 1)
                  .template cast<TypeParam>());
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);
        auto z = asndarray(*y);

        ASSERT_FALSE(z.equals(x));
        ASSERT_EQ(z.base().use_count(), 1);
        ASSERT_EQ(z.shape(), shape_t({1, N}));
        ASSERT_TRUE(allclose(z, x));}

        // 2d: output ndarray is independent of input ndarray.
       {size_t const N(5), M(3);
        auto x = (arange<int>(0, N * M, 1)
                  .reshape({N, M})
                  .template cast<TypeParam>());
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);
        auto z = asndarray(*y);

        ASSERT_FALSE(z.equals(x));
        ASSERT_EQ(z.base().use_count(), 1);
        ASSERT_EQ(z.shape(), shape_t({N, M}));
        ASSERT_TRUE(allclose(z, x));}

        // Transforms possible with expressions.
       {size_t const N(5), M(3);
        auto x = (arange<int>(0, N * M, 1)
                  .reshape({N, M})
                  .template cast<TypeParam>());
        auto y = aseigenarray<TypeParam, mapA_t<TypeParam>>(&x);
        auto z = asndarray(((*y) + TypeParam(2.0)).template cast<TypeParam>());
        auto gt = ((arange<int>(0, N * M, 1) + 2)
                   .reshape({N, M})
                   .template cast<TypeParam>());

        ASSERT_FALSE(z.equals(x));
        ASSERT_EQ(z.base().use_count(), 1);
        ASSERT_EQ(z.shape(), shape_t({N, M}));
        ASSERT_TRUE(allclose(z, gt));}
    }

    REGISTER_TYPED_TEST_CASE_P(TestNdUtilInteropArithmetic,
                               TestIsEigenMappable,
                               TestAsEigenArray,
                               TestAsNdarray);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdUtilInteropArithmetic, MyArithmeticTypes);

    TEST(TestNdUtil, TestOperatorPrint) {
        // soft tests to make sure printed outputs look ok.
        size_t const N(3), M(2);

       {auto x = (nd::arange<int>(0, N * M, 1)
                  .reshape({N, M}));
        std::stringstream res;
        res << x;

        ASSERT_EQ(res.str(), "[[0, 1]\n [2, 3]\n [4, 5]]");}

       {auto x = (nd::arange<float>(0, N * M, 1)
                  .reshape({N, M})) + 0.5;
        std::stringstream res;
        res << x;
        ASSERT_EQ(res.str(), "[[0.5, 1.5]\n [2.5, 3.5]\n [4.5, 5.5]]");}

       {auto x = nd::arange<int>(0, N * M, 1);
        std::stringstream res;
        res << x;
        ASSERT_EQ(res.str(), "[0, 1, 2, 3, 4, 5]");}

       {auto x = (nd::arange<int>(0, N * M, 1)
                  .template cast<cdouble>()) + (0.5 * nd::j<cdouble>());
        std::stringstream res;
        res << x;
        ASSERT_EQ(res.str(), "[(0,0.5), (1,0.5), (2,0.5), (3,0.5), (4,0.5), (5,0.5)]");}
    }
}

#endif // TEST_NDUTIL_CPP
