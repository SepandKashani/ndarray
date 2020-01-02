// ############################################################################
// test_ndfunc.cpp
// ===============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDFUNC_CPP
#define TEST_NDFUNC_CPP

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include "test_type.hpp"

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T> class TestNdFuncBool : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncBool);

    template <typename T> class TestNdFuncInt : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncInt);

    template <typename T> class TestNdFuncSignedInt : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncSignedInt);

    template <typename T> class TestNdFuncFloat : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncFloat);

    template <typename T> class TestNdFuncComplex : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncComplex);

    template <typename T> class TestNdFuncIntFloat : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncIntFloat);

    template <typename T> class TestNdFuncSignedIntFloat : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncSignedIntFloat);

    template <typename T> class TestNdFuncSignedIntFloatComplex : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncSignedIntFloatComplex);

    template <typename T> class TestNdFuncIntFloatComplex : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncIntFloatComplex);

    template <typename T> class TestNdFuncFloatComplex : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncFloatComplex);

    template <typename T> class TestNdFuncArithmetic : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncArithmetic);



    TYPED_TEST_P(TestNdFuncFloat, TestPi) {
        TypeParam const tested = pi<TypeParam>();
        TypeParam const correct = static_cast<TypeParam>(M_PI);

        ASSERT_EQ(tested, correct);
    }

    TYPED_TEST_P(TestNdFuncFloat, TestEuler) {
        TypeParam const tested = e<TypeParam>();
        TypeParam const correct = static_cast<TypeParam>(M_E);

        ASSERT_EQ(tested, correct);
    }

    TYPED_TEST_P(TestNdFuncComplex, TestImaginary) {
        TypeParam const tested = j<TypeParam>();
        TypeParam const correct = TypeParam(0.0, 1.0);

        ASSERT_EQ(tested, correct);
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestRUnderscore) {
       {std::vector<TypeParam> const x;
        auto y = r_(x);
        ASSERT_EQ(y.shape(), shape_t({x.size()}));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& tested_elem = y[{i}];
            TypeParam const& correct_elem = x[i];
            ASSERT_EQ(tested_elem, correct_elem);
        }}

       {std::vector<TypeParam> const x({static_cast<TypeParam>(1),
                                        static_cast<TypeParam>(2),
                                        static_cast<TypeParam>(3)});
        auto y = r_(x);
        ASSERT_EQ(y.shape(), shape_t({x.size()}));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& tested_elem = y[{i}];
            TypeParam const& correct_elem = x[i];
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestZeros) {
        auto x = zeros<TypeParam>({5, 3, 4});
        for(auto const& _x : x) {
            ASSERT_EQ(_x, static_cast<TypeParam>(0.0));
        }
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestOnes) {
        auto x = ones<TypeParam>({5, 3, 4});
        for(auto const& _x : x) {
            ASSERT_EQ(_x, static_cast<TypeParam>(1.0));
        }
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestFull) {
        TypeParam value = static_cast<TypeParam>(3.0);
        auto x = full<TypeParam>({5, 3, 4}, value);
        for(auto const& _x : x) {
            ASSERT_EQ(_x, value);
        }
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestEye) {
        size_t const N = 5;
        auto I = eye<TypeParam>(N);

        ASSERT_EQ(I.shape(), shape_t({N, N}));
        for(size_t i = 0; i < N; ++i) {
            for(size_t j = 0; j < N; ++j) {
                TypeParam const& tested_elem = I[{i, j}];
                TypeParam const correct_elem = ((i == j) ? static_cast<TypeParam>(1.0) : static_cast<TypeParam>(0.0));
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }
    }

    TYPED_TEST_P(TestNdFuncSignedIntFloat, TestArange) {
        // start < stop, positive step
       {TypeParam const start = static_cast<TypeParam>(5.0);
        TypeParam const stop = static_cast<TypeParam>(40.0);
        TypeParam const step = static_cast<TypeParam>(1.25);
        auto x = arange<TypeParam>(start, stop, step);

        auto min_el = std::min_element(x.begin(), x.end());
        auto max_el = std::max_element(x.begin(), x.end());
        ASSERT_EQ(*min_el, start);
        ASSERT_TRUE((*max_el) < stop);
        for(size_t i = 1; i < x.size(); ++i) {
            TypeParam const tested_step = x[{i}] - x[{i - 1}];
            ASSERT_EQ(tested_step, step);
        }}

        // start > stop, negative step
       {TypeParam const start = static_cast<TypeParam>(40.0);
        TypeParam const stop = static_cast<TypeParam>(5.0);
        TypeParam const step = static_cast<TypeParam>(-1.25);
        ndarray<TypeParam> x = arange<TypeParam>(start, stop, step);

        auto min_el = std::min_element(x.begin(), x.end());
        auto max_el = std::max_element(x.begin(), x.end());
        ASSERT_EQ(*max_el, start);
        ASSERT_TRUE((*min_el) > stop);
        for(size_t i = 1; i < x.size(); ++i) {
            TypeParam const tested_step = x[{i}] - x[{i - 1}];
            ASSERT_EQ(tested_step, step);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestLinSpace) {
        size_t const N = 20;

        // start < stop
       {TypeParam const start = static_cast<TypeParam>(5.0);
        TypeParam const stop = static_cast<TypeParam>(40.0);
        auto x = linspace<TypeParam>(start, stop, N);

        auto min_el = std::min_element(x.begin(), x.end());
        auto max_el = std::max_element(x.begin(), x.end());
        ASSERT_TRUE(std::abs(start - (*min_el)) < 1e-5);
        ASSERT_TRUE(std::abs(stop - (*max_el)) < 1e-5);}

        // start > stop
       {TypeParam const start = static_cast<TypeParam>(40.0);
        TypeParam const stop = static_cast<TypeParam>(5.0);
        auto x = linspace<TypeParam>(start, stop, N);

        auto min_el = std::min_element(x.begin(), x.end());
        auto max_el = std::max_element(x.begin(), x.end());
        ASSERT_TRUE(std::abs(start - (*max_el)) < 1e-5);
        ASSERT_TRUE(std::abs(stop - (*min_el)) < 1e-5);}
    }

    TEST(TestNdFunc, TestMeshGrid) {
       { // Empty array
        auto const X = meshgrid(std::vector<ndarray<int>>());
        ASSERT_TRUE(X.size() == 0);}

       { // 1d arrays only
        auto const x = (arange<int>(0, 6, 1)
                        .reshape({2, 3}));
        ASSERT_THROW(meshgrid<int>({x, }), std::runtime_error);}

       { // 1d meshgrid
        auto const x = arange<int>(0, 3, 1);
        auto const y = meshgrid<int>({x, });

        ASSERT_EQ(y.size(), 1u);
        ASSERT_EQ(y[0].shape(), x.shape());
        ASSERT_TRUE(all(y[0] == x, 0)[{0}]);
        ASSERT_TRUE(!y[0].equals(x));}

       { // 2d meshgrid
        auto const x1 = arange<int>(0, 3, 1);
        auto const x2 = arange<int>(3, 7, 1);
        auto const y = meshgrid<int>({x1, x2});

        ASSERT_EQ(y.size(), 2u);
        ASSERT_EQ(y[0].shape(), shape_t({3, 1}));
        ASSERT_EQ(y[1].shape(), shape_t({1, 4}));
        ASSERT_TRUE(allclose(y[0].ravel(), x1));
        ASSERT_TRUE(allclose(y[1].ravel(), x2));
        ASSERT_TRUE(!y[0].equals(x1));
        ASSERT_TRUE(!y[1].equals(x2));}
    }

    TYPED_TEST_P(TestNdFuncSignedIntFloatComplex, TestAbs) {
        // No buffer provided
       {auto x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(-i);
        }

        auto y = abs<TypeParam>(x);

        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(std::abs(*it_x), *it_y);
        }}

        // Buffer provided
       {auto x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(-i);
        }

        auto buffer = zeros<TypeParam>(x.shape());
        auto y = abs<TypeParam>(x, &buffer);

        ASSERT_TRUE(y.equals(buffer));

        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(std::abs(*it_x), *it_y);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloatComplex, TestExp) {
        // No buffer provided
       {auto x = (linspace<double>(-5, 5, 51)
                  .reshape({3, 17})
                  .template cast<TypeParam>());
        auto y = exp(x);

        auto x_ptr = x.begin();
        auto y_ptr = y.begin();
        for(auto x_ptr = x.begin(), y_ptr = y.begin();
            x_ptr != x.end();
            ++x_ptr, ++y_ptr) {
            TypeParam const _ex = std::exp(*x_ptr);
            TypeParam const& _y = *y_ptr;

            ASSERT_EQ(_ex, _y);
        }}

        // Buffer provided
       {auto x = (linspace<double>(-5, 5, 51)
                  .reshape({3, 17})
                  .template cast<TypeParam>());
        auto y = ndarray<TypeParam>(x.shape());

        auto z = exp(x, &y);
        ASSERT_TRUE(y.equals(z));

        auto x_ptr = x.begin();
        auto y_ptr = y.begin();
        for(auto x_ptr = x.begin(), y_ptr = y.begin();
            x_ptr != x.end();
            ++x_ptr, ++y_ptr) {
            TypeParam const _ex = std::exp(*x_ptr);
            TypeParam const& _y = *y_ptr;

            ASSERT_EQ(_ex, _y);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestLog) {
        // No buffer provided
       {auto x = (linspace<double>(0, 5, 51)
                  .reshape({3, 17})
                  .template cast<TypeParam>());
        auto y = log(x);

        auto x_ptr = x.begin();
        auto y_ptr = y.begin();
        for(auto x_ptr = x.begin(), y_ptr = y.begin();
            x_ptr != x.end();
            ++x_ptr, ++y_ptr) {
            TypeParam const _lx = std::log(*x_ptr);
            TypeParam const& _y = *y_ptr;

            ASSERT_EQ(_lx, _y);
        }}

        // Buffer provided
       {auto x = (linspace<double>(0, 5, 51)
                  .reshape({3, 17})
                  .template cast<TypeParam>());
        auto y = ndarray<TypeParam>(x.shape());

        auto z = log(x, &y);
        ASSERT_TRUE(y.equals(z));

        auto x_ptr = x.begin();
        auto y_ptr = y.begin();
        for(auto x_ptr = x.begin(), y_ptr = y.begin();
            x_ptr != x.end();
            ++x_ptr, ++y_ptr) {
            TypeParam const _lx = std::log(*x_ptr);
            TypeParam const& _y = *y_ptr;

            ASSERT_EQ(_lx, _y);
        }}

        // Out-of-Range
       {auto x = r_<TypeParam>({-1.1, -0.1, 0});
        auto y = log(x);
        TypeParam const inf = std::numeric_limits<TypeParam>::infinity();

        ASSERT_TRUE(std::isnan(y[{0}]));
        ASSERT_TRUE(std::isnan(y[{1}]));
        ASSERT_EQ(y[{2}], -inf);}
    }

    TYPED_TEST_P(TestNdFuncBool, TestUniqueBool) {
        auto x = (r_<TypeParam>({  true,  true,  true,  true,
                                   true,  true,  true,  true,
                                   true,  true,  true,  true,

                                   true,  true,  true,  true,
                                   true, false,  true,  true,
                                   true,  true,  true,  true,

                                   true, false,  true,  true,
                                   true,  true,  true,  true,
                                   true,  true,  true,  true,

                                   true,  true, false,  true,
                                   true,  true,  true,  true,
                                   true,  true,  true,  true,

                                   true,  true,  true,  true,
                                   true,  true,  true,  true,
                                   true,  true,  true,  true})
                  .reshape({5, 3, 4}));
        auto gt = r_<TypeParam>({false,  true});

        auto y = unique(x);
        ASSERT_TRUE(all(y == gt, 0)[{0}]);
    }

    TYPED_TEST_P(TestNdFuncInt, TestUniqueInt) {
        auto x = (r_<TypeParam>({ 9, 14, 15, 13,
                                 15,  3, 13,  6,
                                 24, 11,  4, 12,

                                  2,  8,  4, 12,
                                 12,  0, 17, 18,
                                  7, 20, 22, 20,

                                  9,  0, 16, 16,
                                 11, 12, 12, 24,
                                 17,  8, 24,  1,

                                  5, 18,  0, 13,
                                 19, 20, 19, 10,
                                  3,  2, 20, 11,

                                 18, 16, 15, 11,
                                 20, 22, 13,  1,
                                 13, 17,  8,  7})
                  .reshape({5, 3, 4}));
        auto gt = r_<TypeParam>({  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                                  17, 18, 19, 20, 22, 24});

        auto y = unique(x);
        ASSERT_TRUE(all(y == gt, 0)[{0}]);
    }

    TYPED_TEST_P(TestNdFuncSignedInt, TestUniqueSignedInt) {
        auto x = (r_<TypeParam>({ 18,  18, -16,  -8,
                                  19, -22, -20,   5,
                                 -20,  -3,  20,  18,

                                 -21,   4,  13,  16,
                                  22,  11,  12,  10,
                                  17,  -5,  -1,   0,

                                 -15,  16, -17,  19,
                                   0,  18,  -6,  13,
                                   6,  21,   9,   8,

                                 -15,   6, -20,   0,
                                  -3, -15,  12,   4,
                                  10,   8,  13,  -2,

                                   6,  -3,  -6,   6,
                                  -6,  18,  11,   8,
                                   7,   5,   2,  18})
                  .reshape({5, 3, 4}));
        auto gt = r_<TypeParam>({-22, -21, -20, -17, -16, -15,  -8,  -6,  -5,  -3,  -2,  -1,   0,
                                   2,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  16,  17,
                                  18,  19,  20,  21,  22});

        auto y = unique(x);
        ASSERT_TRUE(all(y == gt, 0)[{0}]);
    }

    TYPED_TEST_P(TestNdFuncFloat, TestUniqueFloat) {
        auto x = (r_<double>({-0.28886621, -3.25025675, -0.48248167, -1.03443742,
                              -2.3840612 ,  0.90533714, -2.59267573, -1.52151508,
                               0.43544824, -1.8137003 , -0.73946422, -0.55333563,

                              -1.73222065,  0.19630151,  1.59952943,  0.41002929,
                              -0.78312669,  0.24594309, -0.20868591,  0.36898835,
                              -0.08556923,  0.65485166,  1.07289282, -1.10491782,

                               1.60937945, -1.34027855,  0.14498698, -0.46166759,
                              -0.53442947,  1.61575812,  0.07057698, -0.21031971,
                              -0.1200899 , -0.09881275, -0.80477885,  0.7645095 ,

                               2.33508446,  0.4511968 , -0.03822192,  0.37240465,
                              -0.67925243,  0.89249273,  0.55537163,  0.73655173,
                              -1.52435019,  2.32402258,  1.4468852 ,  1.20810735,

                               1.60602113,  1.14513947, -1.1708768 ,  1.71955045,
                               1.59878618,  1.16292662, -1.22018913,  0.617383  ,
                               0.64868714, -1.12826506,  0.35627699,  0.51074455})
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());
        auto gt = (r_<double>({-3.25025675, -2.59267573, -2.3840612 , -1.8137003 , -1.73222065,
                               -1.52435019, -1.52151508, -1.34027855, -1.22018913, -1.1708768 ,
                               -1.12826506, -1.10491782, -1.03443742, -0.80477885, -0.78312669,
                               -0.73946422, -0.67925243, -0.55333563, -0.53442947, -0.48248167,
                               -0.46166759, -0.28886621, -0.21031971, -0.20868591, -0.1200899 ,
                               -0.09881275, -0.08556923, -0.03822192,  0.07057698,  0.14498698,
                                0.19630151,  0.24594309,  0.35627699,  0.36898835,  0.37240465,
                                0.41002929,  0.43544824,  0.4511968 ,  0.51074455,  0.55537163,
                                0.617383  ,  0.64868714,  0.65485166,  0.73655173,  0.7645095 ,
                                0.89249273,  0.90533714,  1.07289282,  1.14513947,  1.16292662,
                                1.20810735,  1.4468852 ,  1.59878618,  1.59952943,  1.60602113,
                                1.60937945,  1.61575812,  1.71955045,  2.32402258,  2.33508446})
                   .template cast<TypeParam>());

        auto y = unique(x);
        ASSERT_TRUE(allclose(y, gt));
    }

    TYPED_TEST_P(TestNdFuncFloat, TestStd) {
        auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());
        auto gt = (r_<double>({16.97056275, })
                   .broadcast_to({3, 4})
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {auto y = std(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y({1, 3, 4});
        auto z = std(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncComplex, TestStdComplex) {
        auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());
        x += (arange<int>(0, 5 * 3 * 4, 1)
              .reshape({5, 3, 4})
              .template cast<TypeParam>()) * j<TypeParam>();
        auto gt = (r_<int>({24, })
                   .broadcast_to({3, 4})
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {auto y = std(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y({1, 3, 4});
        auto z = std(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncFloatComplex, TestMean) {
        auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());
        auto gt = (arange<int>(24, 36, 1)
                   .reshape({3, 4})
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {auto y = mean(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y({1, 3, 4});
        auto z = mean(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncInt, TestMinInt) { // contains unsigned types
        auto x = (r_<size_t>({18446744073709551599u,                    8u,                   13u, 18446744073709551599u,
                              18446744073709551610u, 18446744073709551609u, 18446744073709551615u, 18446744073709551601u,
                              18446744073709551613u,                    8u,                   25u, 18446744073709551601u,

                                                 6u, 18446744073709551593u,                    2u, 18446744073709551593u,
                              18446744073709551599u,                    3u, 18446744073709551614u, 18446744073709551596u,
                                                17u, 18446744073709551601u,                    1u,                   12u,

                                                 9u, 18446744073709551602u, 18446744073709551602u,                    3u,
                                                 2u, 18446744073709551591u, 18446744073709551612u,                   23u,
                                                22u, 18446744073709551594u,                    9u,                    9u,

                                                11u,                   12u,                   16u, 18446744073709551604u,
                                                 1u, 18446744073709551615u, 18446744073709551596u, 18446744073709551596u,
                              18446744073709551615u, 18446744073709551594u, 18446744073709551607u,                   24u,

                                                 4u, 18446744073709551608u, 18446744073709551595u, 18446744073709551615u,
                                                14u, 18446744073709551610u, 18446744073709551603u,                   19u,
                              18446744073709551608u,                    7u,                   22u,                   16u})
                  .reshape({5, 3, 4}));
        auto gt = (r_<size_t>({                    4u,                    8u,                    2u,                    3u,
                                                   1u,                    3u, 18446744073709551596u,                   19u,
                                                  17u,                    7u,                    1u,                    9u})
                   .reshape({3, 4}));

        // No buffer provided, keepdims
       {auto y = min(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<size_t> y({1, 3, 4});
        auto z = min(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));};
    }

    TYPED_TEST_P(TestNdFuncSignedInt, TestMinSignedInt) {
        auto x = (r_<int>({-17,   8,  13, -17,
                            -6,  -7,  -1, -15,
                            -3,   8,  25, -15,

                             6, -23,   2, -23,
                           -17,   3,  -2, -20,
                            17, -15,   1,  12,

                             9, -14, -14,   3,
                             2, -25,  -4,  23,
                            22, -22,   9,   9,

                            11,  12,  16, -12,
                             1,  -1, -20, -20,
                            -1, -22,  -9,  24,

                             4,  -8, -21,  -1,
                            14,  -6, -13,  19,
                            -8,   7,  22,  16})
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());
        auto gt = (r_<int>({-17, -23, -21, -23,
                            -17, -25, -20, -20,
                             -8, -22,  -9, -15})
                   .reshape({3, 4})
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {auto y = min(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y({1, 3, 4});
        auto z = min(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));};
    }

    TYPED_TEST_P(TestNdFuncFloat, TestMinFloat) {
        auto x = (r_<double>({-0.28886621, -3.25025675, -0.48248167, -1.03443742,
                              -2.3840612 ,  0.90533714, -2.59267573, -1.52151508,
                               0.43544824, -1.8137003 , -0.73946422, -0.55333563,

                              -1.73222065,  0.19630151,  1.59952943,  0.41002929,
                              -0.78312669,  0.24594309, -0.20868591,  0.36898835,
                              -0.08556923,  0.65485166,  1.07289282, -1.10491782,

                               1.60937945, -1.34027855,  0.14498698, -0.46166759,
                              -0.53442947,  1.61575812,  0.07057698, -0.21031971,
                              -0.1200899 , -0.09881275, -0.80477885,  0.7645095 ,

                               2.33508446,  0.4511968 , -0.03822192,  0.37240465,
                              -0.67925243,  0.89249273,  0.55537163,  0.73655173,
                              -1.52435019,  2.32402258,  1.4468852 ,  1.20810735,

                               1.60602113,  1.14513947, -1.1708768 ,  1.71955045,
                               1.59878618,  1.16292662, -1.22018913,  0.617383  ,
                               0.64868714, -1.12826506,  0.35627699,  0.51074455})
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());
        auto gt = (r_<double>({-1.73222065, -3.25025675, -1.1708768 , -1.03443742,
                               -2.3840612 ,  0.24594309, -2.59267573, -1.52151508,
                               -1.52435019, -1.8137003 , -0.80477885, -1.10491782})
                   .reshape({3, 4})
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {auto y = min(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y({1, 3, 4});
        auto z = min(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncInt, TestMaxInt) { // contains unsigned types
        auto x = (r_<size_t>({18446744073709551599u,                    8u,                   13u, 18446744073709551599u,
                              18446744073709551610u, 18446744073709551609u, 18446744073709551615u, 18446744073709551601u,
                              18446744073709551613u,                    8u,                   25u, 18446744073709551601u,

                                                 6u, 18446744073709551593u,                    2u, 18446744073709551593u,
                              18446744073709551599u,                    3u, 18446744073709551614u, 18446744073709551596u,
                                                17u, 18446744073709551601u,                    1u,                   12u,

                                                 9u, 18446744073709551602u, 18446744073709551602u,                    3u,
                                                 2u, 18446744073709551591u, 18446744073709551612u,                   23u,
                                                22u, 18446744073709551594u,                    9u,                    9u,

                                                11u,                   12u,                   16u, 18446744073709551604u,
                                                 1u, 18446744073709551615u, 18446744073709551596u, 18446744073709551596u,
                              18446744073709551615u, 18446744073709551594u, 18446744073709551607u,                   24u,

                                                 4u, 18446744073709551608u, 18446744073709551595u, 18446744073709551615u,
                                                14u, 18446744073709551610u, 18446744073709551603u,                   19u,
                              18446744073709551608u,                    7u,                   22u,                   16u})
                  .reshape({5, 3, 4}));
        auto gt = (r_<size_t>({18446744073709551599u, 18446744073709551608u, 18446744073709551602u, 18446744073709551615u,
                               18446744073709551610u, 18446744073709551615u, 18446744073709551615u, 18446744073709551601u,
                               18446744073709551615u, 18446744073709551601u, 18446744073709551607u, 18446744073709551601u})
                   .reshape({3, 4}));

        // No buffer provided, keepdims
       {auto y = max(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<size_t> y({1, 3, 4});
        auto z = max(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));};
    }

    TYPED_TEST_P(TestNdFuncSignedInt, TestMaxSignedInt) {
        auto x = (r_<int>({-17,   8,  13, -17,
                            -6,  -7,  -1, -15,
                            -3,   8,  25, -15,

                             6, -23,   2, -23,
                           -17,   3,  -2, -20,
                            17, -15,   1,  12,

                             9, -14, -14,   3,
                             2, -25,  -4,  23,
                            22, -22,   9,   9,

                            11,  12,  16, -12,
                             1,  -1, -20, -20,
                            -1, -22,  -9,  24,

                             4,  -8, -21,  -1,
                            14,  -6, -13,  19,
                            -8,   7,  22,  16})
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());
        auto gt = (r_<int>({11, 12, 16,  3,
                            14,  3, -1, 23,
                            22,  8, 25, 24})
                   .reshape({3, 4})
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {auto y = max(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y({1, 3, 4});
        auto z = max(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));};
    }

    TYPED_TEST_P(TestNdFuncFloat, TestMaxFloat) {
        auto x = (r_<double>({-0.28886621, -3.25025675, -0.48248167, -1.03443742,
                              -2.3840612 ,  0.90533714, -2.59267573, -1.52151508,
                               0.43544824, -1.8137003 , -0.73946422, -0.55333563,

                              -1.73222065,  0.19630151,  1.59952943,  0.41002929,
                              -0.78312669,  0.24594309, -0.20868591,  0.36898835,
                              -0.08556923,  0.65485166,  1.07289282, -1.10491782,

                               1.60937945, -1.34027855,  0.14498698, -0.46166759,
                              -0.53442947,  1.61575812,  0.07057698, -0.21031971,
                              -0.1200899 , -0.09881275, -0.80477885,  0.7645095 ,

                               2.33508446,  0.4511968 , -0.03822192,  0.37240465,
                              -0.67925243,  0.89249273,  0.55537163,  0.73655173,
                              -1.52435019,  2.32402258,  1.4468852 ,  1.20810735,

                               1.60602113,  1.14513947, -1.1708768 ,  1.71955045,
                               1.59878618,  1.16292662, -1.22018913,  0.617383  ,
                               0.64868714, -1.12826506,  0.35627699,  0.51074455})
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());
        auto gt = (r_<double>({2.33508446, 1.14513947, 1.59952943, 1.71955045,
                               1.59878618, 1.61575812, 0.55537163, 0.73655173,
                               0.64868714, 2.32402258, 1.4468852 , 1.20810735})
                   .reshape({3, 4})
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {auto y = max(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y({1, 3, 4});
        auto z = max(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestCeil) {
        auto x = (arange<TypeParam>(-3.1, 3.9, 0.5)
                  .reshape({2, 7}));
        auto gt = (r_(std::vector<TypeParam> {-3, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4})
                   .reshape({2, 7}));

        // No buffer provided
       {auto y = ceil<TypeParam>(x);
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = gt.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        auto _y = ceil<TypeParam>(x, &y);
        ASSERT_TRUE(_y.equals(y));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = gt.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestFloor) {
        auto x = (arange<TypeParam>(-3.1, 3.9, 0.5)
                  .reshape({2, 7}));
        auto gt = (r_(std::vector<TypeParam> {-4, -3, -3, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 3})
                   .reshape({2, 7}));

        // No buffer provided
       {auto y = floor<TypeParam>(x);
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = gt.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        auto _y = floor<TypeParam>(x, &y);
        ASSERT_TRUE(_y.equals(y));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = gt.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}
    }

    TYPED_TEST_P(TestNdFuncIntFloat, TestClip) {
        auto x = (arange<int>(3, 9, 1)
                  .reshape({2, 3})
                  .template cast<TypeParam>());
        TypeParam const down = static_cast<TypeParam>(5);
        TypeParam const up = static_cast<TypeParam>(7);
        auto gt = (r_(std::vector<TypeParam> {5, 5, 5, 6, 7, 7})
                   .reshape({2, 3}));

        // No buffer provided
       {auto y = clip<TypeParam>(x, down, up);
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = gt.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        auto _y = clip<TypeParam>(x, down, up, &y);
        ASSERT_TRUE(_y.equals(y));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = gt.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}
    }

    TYPED_TEST_P(TestNdFuncSignedIntFloat, TestSign) {
        auto x = (arange<TypeParam>(-3, 3, 1)
                  .reshape({2, 3}));
        auto gt = (r_(std::vector<TypeParam> {-1, -1, -1, 0, 1, 1})
                   .reshape({2, 3}));

        // No buffer provided
       {auto y = sign<TypeParam>(x);
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = gt.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        auto _y = sign<TypeParam>(x, &y);
        ASSERT_TRUE(_y.equals(y));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = gt.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestIsClose) {
        // No buffer provided
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        auto y = isclose(x, x);
        for(auto it_y = y.begin(); it_y != y.end(); ++it_y) {
            ASSERT_TRUE(*it_y);
        }}

        // Buffer provided
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        ndarray<bool> y(x.shape());
        auto z = isclose(x, x, &y);
        ASSERT_TRUE(y.equals(z));

        for(auto it_y = y.begin(); it_y != y.end(); ++it_y) {
            ASSERT_TRUE(*it_y);
        }}
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestAllClose) {
        auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        bool y = allclose(x, x);
        ASSERT_TRUE(y);
    }



    /* Reduction Functions ================================================= */
    TYPED_TEST_P(TestNdFuncIntFloatComplex, TestSum) {
        // No buffer provided, keepdims
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        auto y = sum(x, 0);
        ASSERT_EQ(y.shape(), shape_t({3, 4}));
        TypeParam correct_elem = static_cast<TypeParam>(120.0);
        for(auto it_y = y.begin(); it_y != y.end();
            ++it_y, correct_elem += static_cast<TypeParam>(5.0)) {
            ASSERT_EQ(*it_y, correct_elem);
        }}

        // Buffer provided, dropdims
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        ndarray<TypeParam> y({1, 3, 4});
        auto z = sum(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));

        TypeParam correct_elem = static_cast<TypeParam>(120.0);
        for(auto it_y = y.begin(); it_y != y.end();
            ++it_y, correct_elem += static_cast<TypeParam>(5.0)) {
            ASSERT_EQ(*it_y, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdFuncIntFloatComplex, TestProd) {
        auto x = (arange<int>(0, 2 * 3 * 4, 1)
                  .reshape({2, 3, 4})
                  .template cast<TypeParam>());
        auto gt = (r_<int>({  0,  13,  28,  45,
                             64,  85, 108, 133,
                            160, 189, 220, 253})
                   .reshape({3, 4})
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {auto y = prod(x, 0);
        ASSERT_TRUE(allclose(y, gt));}

        // Buffer provided, dropdims
       {ndarray<TypeParam> y({1, 3, 4});
        auto z = prod(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TEST(TestNdFunc, TestStack) {
        { // 1d arrays
            auto const x1 = arange<int>(0, 6, 1);
            auto const x2 = arange<int>(6, 12, 1) * 2;
            auto const x3 = arange<int>(12, 18, 1) * 3;
            auto const x4 = arange<int>(18, 24, 1) * 4;
            auto const x_broken = arange<int>(0, 3, 1);

           { // Same shapes
            ASSERT_NO_THROW(stack(std::vector({x1,}), 0));
            ASSERT_NO_THROW(stack(std::vector({x1, x2, x3, x4,}), 0));
            ASSERT_THROW(stack(std::vector({x_broken, x1, x2, x3, x4}), 0), std::runtime_error);
            ASSERT_THROW(stack(std::vector({x1, x_broken, x2, x3, x4}), 0), std::runtime_error);
            ASSERT_THROW(stack(std::vector({x1, x2, x_broken, x3, x4}), 0), std::runtime_error);
            ASSERT_THROW(stack(std::vector({x1, x2, x3, x_broken, x4}), 0), std::runtime_error);
            ASSERT_THROW(stack(std::vector({x1, x2, x3, x4, x_broken}), 0), std::runtime_error);}

           { // axis
            ASSERT_NO_THROW(stack(std::vector({x1,}), 0));
            ASSERT_NO_THROW(stack(std::vector({x1,}), 1));
            ASSERT_THROW(stack(std::vector({x1,}), 2), std::runtime_error);
            ASSERT_NO_THROW(stack(std::vector({x1, x2, x3, x4,}), 0));
            ASSERT_NO_THROW(stack(std::vector({x1, x2, x3, x4,}), 1));
            ASSERT_THROW(stack(std::vector({x1, x2, x3, x4,}), 2), std::runtime_error);}

           { // correct output (single)
           {auto gt = x1.reshape({1, 6});
            auto y = stack(std::vector({x1, }), 0);
            ASSERT_TRUE(all(gt == y, 0)[{0}]);}
           {auto gt = x1.reshape({6, 1});
            auto y = stack(std::vector({x1, }), 1);
            ASSERT_TRUE(all(gt == y, 0)[{0}]);}}

           { // correct output (multi)
           {auto gt = (r_<int>({ 0,  1,  2,  3,  4,  5,
                                12, 14, 16, 18, 20, 22,
                                36, 39, 42, 45, 48, 51,
                                72, 76, 80, 84, 88, 92})
                       .reshape({4, 6}));
            auto y = stack(std::vector({x1, x2, x3, x4}), 0);
            ASSERT_TRUE( allclose(gt, y));}
           {auto gt = (r_<int>({ 0, 12, 36, 72,
                                 1, 14, 39, 76,
                                 2, 16, 42, 80,
                                 3, 18, 45, 84,
                                 4, 20, 48, 88,
                                 5, 22, 51, 92})
                               .reshape({6, 4}));
            auto y = stack(std::vector({x1, x2, x3, x4}), 1);
            ASSERT_TRUE( allclose(gt, y));}}

           { // output array provided
            auto gt = (r_<int>({ 0,  1,  2,  3,  4,  5,
                                12, 14, 16, 18, 20, 22,
                                36, 39, 42, 45, 48, 51,
                                72, 76, 80, 84, 88, 92})
                       .reshape({4, 6}));
            ndarray<int> y(gt.shape());
            auto z = stack(std::vector({x1, x2, x3, x4}), 0, &y);
            ASSERT_TRUE(y.equals(z));
            ASSERT_TRUE(allclose(gt, y));
           }
        }

        { // 2d arrays
            shape_t const shape({2, 3});
            auto const x1 = arange<int>(0, 6, 1).reshape(shape);
            auto const x2 = arange<int>(6, 12, 1).reshape(shape) * 2;
            auto const x3 = arange<int>(12, 18, 1).reshape(shape) * 3;
            auto const x4 = arange<int>(18, 24, 1).reshape(shape) * 4;
            auto const x_broken = arange<int>(0, 3, 1);

           { // Same shapes
            ASSERT_NO_THROW(stack(std::vector({x1,}), 0));
            ASSERT_NO_THROW(stack(std::vector({x1, x2, x3, x4,}), 0));
            ASSERT_THROW(stack(std::vector({x_broken, x1, x2, x3, x4}), 0), std::runtime_error);
            ASSERT_THROW(stack(std::vector({x1, x_broken, x2, x3, x4}), 0), std::runtime_error);
            ASSERT_THROW(stack(std::vector({x1, x2, x_broken, x3, x4}), 0), std::runtime_error);
            ASSERT_THROW(stack(std::vector({x1, x2, x3, x_broken, x4}), 0), std::runtime_error);
            ASSERT_THROW(stack(std::vector({x1, x2, x3, x4, x_broken}), 0), std::runtime_error);}

           { // axis
            ASSERT_NO_THROW(stack(std::vector({x1,}), 0));
            ASSERT_NO_THROW(stack(std::vector({x1,}), 1));
            ASSERT_NO_THROW(stack(std::vector({x1,}), 2));
            ASSERT_THROW(stack(std::vector({x1,}), 3), std::runtime_error);
            ASSERT_NO_THROW(stack(std::vector({x1, x2, x3, x4,}), 0));
            ASSERT_NO_THROW(stack(std::vector({x1, x2, x3, x4,}), 1));
            ASSERT_NO_THROW(stack(std::vector({x1, x2, x3, x4,}), 2));
            ASSERT_THROW(stack(std::vector({x1, x2, x3, x4,}), 3), std::runtime_error);}

           { // correct output (single)
           {auto gt = x1.reshape({1, 2, 3});
            auto y = stack(std::vector({x1,}), 0);
            ASSERT_TRUE( allclose(gt, y));}
           {auto gt = x1.reshape({2, 1, 3});
            auto y = stack(std::vector({x1,}), 1);
            ASSERT_TRUE( allclose(gt, y));}
           {auto gt = x1.reshape({2, 3, 1});
            auto y = stack(std::vector({x1,}), 2);
            ASSERT_TRUE( allclose(gt, y));}}

           { // correct output (multi)
           {auto gt = (r_<int>({ 0,  1,  2,
                                 3,  4,  5,

                                12, 14, 16,
                                18, 20, 22,

                                36, 39, 42,
                                45, 48, 51,

                                72, 76, 80,
                                84, 88, 92})
                       .reshape({4, 2, 3}));
            auto y = stack(std::vector({x1, x2, x3, x4}), 0);
            ASSERT_TRUE( allclose(gt, y));}
           {auto gt = (r_<int>({ 0,  1,  2,
                                12, 14, 16,
                                36, 39, 42,
                                72, 76, 80,

                                 3,  4,  5,
                                18, 20, 22,
                                45, 48, 51,
                                84, 88, 92})
                       .reshape({2, 4, 3}));
            auto y = stack(std::vector({x1, x2, x3, x4}), 1);
            ASSERT_TRUE( allclose(gt, y));}
           {auto gt = (r_<int>({ 0, 12, 36, 72,
                                 1, 14, 39, 76,
                                 2, 16, 42, 80,

                                 3, 18, 45, 84,
                                 4, 20, 48, 88,
                                 5, 22, 51, 92})
                       .reshape({2, 3, 4}));
            auto y = stack(std::vector({x1, x2, x3, x4}), 2);
            ASSERT_TRUE( allclose(gt, y));}}

           { // output array provided
            auto gt = (r_<int>({ 0,  1,  2,
                                 3,  4,  5,

                                12, 14, 16,
                                18, 20, 22,

                                36, 39, 42,
                                45, 48, 51,

                                72, 76, 80,
                                84, 88, 92})
                       .reshape({4, 2, 3}));
            ndarray<int> y(gt.shape());
            auto z = stack(std::vector({x1, x2, x3, x4}), 0, &y);
            ASSERT_TRUE(y.equals(z));
            ASSERT_TRUE( allclose(gt, y));}
        }
    }

    TEST(TestNdFunc, TestAny) {
        // No buffer provided, keepdims
       {auto x = zeros<bool>({2, 3});
        x[{1, 1}] = true;

        auto y = any(x, 0);
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.data()[0],  false);
        ASSERT_EQ(y.data()[1],  true);
        ASSERT_EQ(y.data()[2],  false);}

       {auto x = zeros<bool>({2, 3});
        x[{1, 1}] = true;

        ndarray<bool> y(shape_t({1, 3}));
        auto z = any(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_EQ(y.shape(), shape_t({1, 3}));
        ASSERT_EQ(y.data()[0],  false);
        ASSERT_EQ(y.data()[1],  true);
        ASSERT_EQ(y.data()[2],  false);}
    }

    TEST(TestNdFunc, TestAll) {
        // No buffer provided, keepdims
       {auto x = zeros<bool>({2, 3});
        x[{1, 1}] = true;

        auto y = all(x, 0);
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.data()[0],  false);
        ASSERT_EQ(y.data()[1],  false);
        ASSERT_EQ(y.data()[2],  false);}

       {auto x = zeros<bool>({2, 3});
        x[{1, 1}] = true;

        ndarray<bool> y(shape_t({1, 3}));
        auto z = all(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_EQ(y.shape(), shape_t({1, 3}));
        ASSERT_EQ(y.data()[0],  false);
        ASSERT_EQ(y.data()[1],  false);
        ASSERT_EQ(y.data()[2],  false);}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestSin) {
        // No buffer provided
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        auto y = sin(x);
        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::sin(*it_x));
        }}

        // Buffer provided
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = sin(x, &y);
        ASSERT_TRUE(y.equals(z));

        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::sin(*it_x));
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestCos) {
        // No buffer provided
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        auto y = cos(x);
        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::cos(*it_x));
        }}

        // Buffer provided
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = cos(x, &y);
        ASSERT_TRUE(y.equals(z));

        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::cos(*it_x));
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestTan) {
        // No buffer provided
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        auto y = tan(x);
        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::tan(*it_x));
        }}

        // Buffer provided
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4})
                  .template cast<TypeParam>());

        ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = tan(x, &y);
        ASSERT_TRUE(y.equals(z));

        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::tan(*it_x));
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestArcSin) {
        auto x = (r_<TypeParam>({0, 1/M_SQRT2, 1, 1/M_SQRT2, 0, -1/M_SQRT2, -1, -1/M_SQRT2})
                  .reshape({2, 4}));
        auto gt = (r_<TypeParam>({0, M_PI/4, M_PI/2, M_PI/4, 0, -M_PI/4, -M_PI/2, -M_PI/4})
                   .reshape({2, 4}));

        // No buffer provided
       {ndarray<TypeParam> y = arcsin(x);
        ASSERT_TRUE(allclose(y, gt));}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        auto z = arcsin(x, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}

        // Out-of-Range
       {auto x = r_<TypeParam>({-1.1, 1.1});
        auto y = arcsin(x);
        ASSERT_TRUE(std::isnan(y[{0}]));
        ASSERT_TRUE(std::isnan(y[{1}]));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestArcCos) {
        auto x = (r_<TypeParam>({1, 1/M_SQRT2, 0, -1/M_SQRT2, -1, -1/M_SQRT2, 0, 1/M_SQRT2})
                  .reshape({2, 4}));
        auto gt = (r_<TypeParam>({0, M_PI/4, M_PI/2, 3*M_PI/4, M_PI, 3*M_PI/4, M_PI/2, M_PI/4})
                   .reshape({2, 4}));

        // No buffer provided
       {auto y = arccos(x);
        ASSERT_TRUE(allclose(y, gt));}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        auto z = arccos(x, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}

        // Out-of-Range
       {auto x = r_<TypeParam>({-1.1, 1.1});
        auto y = arccos(x);
        ASSERT_TRUE(std::isnan(y[{0}]));
        ASSERT_TRUE(std::isnan(y[{1}]));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestArcTan) {
        TypeParam const inf = std::numeric_limits<TypeParam>::infinity();
        auto x = (r_<TypeParam>({0, 1, inf, -1, 0, 1, -inf, -1})
                  .reshape({2, 4}));
        auto gt = (r_<TypeParam>({0, M_PI/4, M_PI/2, -M_PI/4, 0, M_PI/4, -M_PI/2, -M_PI/4})
                   .reshape({2, 4}));

        // No buffer provided
       {auto y = arctan(x);
        ASSERT_TRUE(allclose(y, gt));}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        auto z = arctan(x, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestArcTan2) {
        auto x = (r_<TypeParam>({1, 1/M_SQRT2, 0, -1/M_SQRT2, -1, -1/M_SQRT2, 0, 1/M_SQRT2})
                  .reshape({2, 4}));
        auto y = (r_<TypeParam>({0, 1/M_SQRT2, 1, 1/M_SQRT2, 0, -1/M_SQRT2, -1, -1/M_SQRT2})
                  .reshape({2, 4}));
        auto gt = (r_<TypeParam>({0, M_PI/4, M_PI/2, 3*M_PI/4, M_PI, -3*M_PI/4, -M_PI/2, -M_PI/4})
                   .reshape({2, 4}));

        // No buffer provided
       {auto res = arctan2(y, x);
        ASSERT_TRUE(allclose(res, gt));}

        // Buffer provided
       {ndarray<TypeParam> res(shape_t({2, 4}));
        auto z = arctan2(y, x, &res);
        ASSERT_TRUE(res.equals(z));
        ASSERT_TRUE(allclose(res, gt));}

        // Broadcasting
       {TypeParam const sqrt_3 = std::sqrt(3.0);
        auto x = r_<TypeParam>({1, });
        auto y = r_<TypeParam>({0, 1/sqrt_3, 1, -1/sqrt_3, -1});
        auto gt = r_<TypeParam>({0, M_PI/6, M_PI/4, -M_PI/6, -M_PI/4});

        auto res = arctan2(y, x);
        ASSERT_TRUE(allclose(res, gt));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestDeg2Rad) {
        auto ratio = r_(std::vector<TypeParam>({0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75}));
        auto deg = ratio * ndarray<TypeParam>(180.0);
        auto rad_gt = ratio * ndarray<TypeParam>(M_PI);

        // No buffer provided
       {auto rad = deg2rad<TypeParam>(deg);
        ASSERT_TRUE(allclose(rad, rad_gt));}

        // Buffer provided
       {ndarray<TypeParam> rad(deg.shape());
        auto _rad = deg2rad<TypeParam>(deg, &rad);
        ASSERT_TRUE(_rad.equals(rad));
        ASSERT_TRUE(allclose(rad, rad_gt));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestRad2Deg) {
        auto ratio = r_(std::vector<TypeParam>({0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75}));
        auto rad = ratio * ndarray<TypeParam>(M_PI);
        auto deg_gt = ratio * ndarray<TypeParam>(180.0);

        // No buffer provided
       {auto deg = rad2deg<TypeParam>(rad);
        ASSERT_TRUE(allclose(deg, deg_gt));}

        // Buffer provided
       {ndarray<TypeParam> deg(rad.shape());
        auto _deg = rad2deg<TypeParam>(rad, &deg);
        ASSERT_TRUE(_deg.equals(deg));
        ASSERT_TRUE(allclose(deg, deg_gt));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestSinc) {
        auto x = r_<TypeParam>({0.        , 0.05263158, 0.10526316, 0.15789474, 0.21052632,
                                0.26315789, 0.31578947, 0.36842105, 0.42105263, 0.47368421,
                                0.52631579, 0.57894737, 0.63157895, 0.68421053, 0.73684211,
                                0.78947368, 0.84210526, 0.89473684, 0.94736842, 1. });
        auto gt = r_<TypeParam>({1.00000000e+00, 9.95449621e-01, 9.81872985e-01, 9.59492150e-01,
                                 9.28672399e-01, 8.89915138e-01, 8.43848160e-01, 7.91213481e-01,
                                 7.32853010e-01, 6.69692359e-01, 6.02723123e-01, 5.32984007e-01,
                                 4.61541197e-01, 3.89468382e-01, 3.17826835e-01, 2.47645973e-01,
                                 1.79904778e-01, 1.15514469e-01, 5.53027567e-02, 3.89817183e-17});

        // No buffer provided
       {auto y = sinc(x);
        ASSERT_TRUE(allclose(y, gt, 1e-5, 1e-7));}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        auto z = sinc(x, &y);
        ASSERT_TRUE(z.equals(y));
        ASSERT_TRUE(allclose(y, gt, 1e-5, 1e-7));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestReal) {
        // Contiguous array
       {ndarray<std::complex<TypeParam>> x({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                 -static_cast<TypeParam>(i + 1.0));
        }

        auto y = real(x);
        ASSERT_EQ(y.shape(), x.shape());

        auto it_y = y.begin();
        for(auto it_x = x.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).real(), *it_y);
        }}

        // Non-contiguous array
       {ndarray<std::complex<TypeParam>> _x({5, 3, 8});
        for(size_t i = 0; i < _x.size(); ++i) {
            _x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                  -static_cast<TypeParam>(i + 1.0));
        }
        auto x = _x({util::slice(), util::slice(), util::slice(0, 8, 2)});

        auto y = real(x);
        ASSERT_EQ(y.shape(), x.shape());

        auto it_y = y.begin();
        for(auto it_x = x.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).real(), *it_y);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestImag) {
        // Contiguous array
       {ndarray<std::complex<TypeParam>> x({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                 -static_cast<TypeParam>(i + 1.0));
        }

        auto y = imag(x);
        ASSERT_EQ(y.shape(), x.shape());

        auto it_y = y.begin();
        for(auto it_x = x.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).imag(), *it_y);
        }}

        // Non-contiguous array
       {ndarray<std::complex<TypeParam>> _x({5, 3, 8});
        for(size_t i = 0; i < _x.size(); ++i) {
            _x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                  -static_cast<TypeParam>(i + 1.0));
        }
        auto x = _x({util::slice(), util::slice(), util::slice(0, 8, 2)});

        auto y = imag(x);
        ASSERT_EQ(y.shape(), x.shape());

        auto it_y = y.begin();
        for(auto it_x = x.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).imag(), *it_y);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestConj) {
        // No buffer provided
       {ndarray<std::complex<TypeParam>> x({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                 -static_cast<TypeParam>(i + 1.0));
        }

        auto y = conj(x);
        ASSERT_EQ(y.shape(), x.shape());

        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).real(), (*it_y).real());
            ASSERT_EQ((*it_x).imag(), -(*it_y).imag());
        }}

        // Buffer provided
       {ndarray<std::complex<TypeParam>> x({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                 -static_cast<TypeParam>(i + 1.0));
        }

        ndarray<std::complex<TypeParam>> buffer(x.shape());
        auto y = conj(x, &buffer);
        ASSERT_TRUE(buffer.equals(y));

        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).real(), (*it_y).real());
            ASSERT_EQ((*it_x).imag(), -(*it_y).imag());
        }}
    }



    // Initialize tests ====================================================
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncBool,
                               TestUniqueBool);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncBool, MyBoolTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncInt,
                               TestUniqueInt,
                               TestMinInt,
                               TestMaxInt);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncInt, MyIntTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncSignedInt,
                               TestUniqueSignedInt,
                               TestMinSignedInt,
                               TestMaxSignedInt);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncSignedInt, MySignedIntTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncFloat,
                               TestPi,
                               TestEuler,
                               TestLinSpace,
                               TestUniqueFloat,
                               TestLog,
                               TestMinFloat,
                               TestMaxFloat,
                               TestCeil,
                               TestFloor,
                               TestSin,
                               TestCos,
                               TestTan,
                               TestArcSin,
                               TestArcCos,
                               TestArcTan,
                               TestArcTan2,
                               TestDeg2Rad,
                               TestRad2Deg,
                               TestSinc,
                               TestStd,
                               TestReal,
                               TestImag,
                               TestConj);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncFloat, MyFloatTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncComplex,
                               TestImaginary,
                               TestStdComplex);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncComplex, MyComplexTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncIntFloat,
                               TestClip);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncIntFloat, MyIntFloatTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncSignedIntFloat,
                               TestArange,
                               TestSign);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncSignedIntFloat, MySignedIntFloatTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncSignedIntFloatComplex,
                               TestAbs);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncSignedIntFloatComplex, MySignedIntFloatComplexTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncIntFloatComplex,
                               TestSum,
                               TestProd);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncIntFloatComplex, MyIntFloatComplexTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncFloatComplex,
                               TestExp,
                               TestMean);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncFloatComplex, MyFloatComplexTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncArithmetic,
                               TestRUnderscore,
                               TestZeros,
                               TestOnes,
                               TestFull,
                               TestEye,
                               TestIsClose,
                               TestAllClose);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncArithmetic, MyArithmeticTypes);
}

#endif // TEST_NDFUNC_CPP
