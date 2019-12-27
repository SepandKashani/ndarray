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

    template <typename T> class TestNdFuncBoolInt : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncBoolInt);

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

    template <typename T> class TestNdFuncSignedArithmetic : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncSignedArithmetic);

    template <typename T> class TestNdFuncArithmetic : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncArithmetic);



    /* Constants =========================================================== */
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



    /* Utility ============================================================= */
    TYPED_TEST_P(TestNdFuncArithmetic, TestRUnderscore) {
       {std::vector<TypeParam> const x;
        ndarray<TypeParam> y = r_(x);
        ASSERT_EQ(y.shape(), shape_t({x.size()}));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& tested_elem = y[{i}];
            TypeParam const& correct_elem = x[i];
            ASSERT_EQ(tested_elem, correct_elem);
        }}

       {std::vector<TypeParam> const x({static_cast<TypeParam>(1),
                                        static_cast<TypeParam>(2),
                                        static_cast<TypeParam>(3)});
        ndarray<TypeParam> y = r_(x);
        ASSERT_EQ(y.shape(), shape_t({x.size()}));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& tested_elem = y[{i}];
            TypeParam const& correct_elem = x[i];
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestZeros) {
        ndarray<TypeParam> x = zeros<TypeParam>(shape_t({5, 3, 4}));
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, static_cast<TypeParam>(0.0));
        }
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestOnes) {
        ndarray<TypeParam> x = ones<TypeParam>(shape_t({5, 3, 4}));
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, static_cast<TypeParam>(1.0));
        }
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestFull) {
        TypeParam value = static_cast<TypeParam>(3.0);
        ndarray<TypeParam> x = full<TypeParam>(shape_t({5, 3, 4}), value);
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, value);
        }
    }

    TYPED_TEST_P(TestNdFuncArithmetic, TestEye) {
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

    TYPED_TEST_P(TestNdFuncSignedIntFloat, TestArange) {
        // start < stop, positive step
       {TypeParam const start = static_cast<TypeParam>(5.0);
        TypeParam const stop = static_cast<TypeParam>(40.0);
        TypeParam const step = static_cast<TypeParam>(1.25);
        ndarray<TypeParam> x = arange<TypeParam>(start, stop, step);

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
        ndarray<TypeParam> x = linspace<TypeParam>(start, stop, N);

        auto min_el = std::min_element(x.begin(), x.end());
        auto max_el = std::max_element(x.begin(), x.end());
        ASSERT_TRUE(std::abs(start - (*min_el)) < 1e-5);
        ASSERT_TRUE(std::abs(stop - (*max_el)) < 1e-5);}

        // start > stop
       {TypeParam const start = static_cast<TypeParam>(40.0);
        TypeParam const stop = static_cast<TypeParam>(5.0);
        ndarray<TypeParam> x = linspace<TypeParam>(start, stop, N);

        auto min_el = std::min_element(x.begin(), x.end());
        auto max_el = std::max_element(x.begin(), x.end());
        ASSERT_TRUE(std::abs(start - (*max_el)) < 1e-5);
        ASSERT_TRUE(std::abs(stop - (*min_el)) < 1e-5);}
    }



    /* Math Functions ====================================================== */
    TYPED_TEST_P(TestNdFuncSignedIntFloatComplex, TestAbs) {
        // No buffer provided
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
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
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(-i);
        }

        ndarray<TypeParam> buffer = zeros<TypeParam>(x.shape());
        ndarray<TypeParam> y = abs<TypeParam>(x, &buffer);

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

        ndarray<TypeParam> z = exp(x, &y);
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

        ndarray<TypeParam> z = log(x, &y);
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

    TYPED_TEST_P(TestNdFuncFloat, TestStd) {
        ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());
        ndarray<TypeParam> gt = (r_<double>({16.97056275, })
                                 .broadcast_to(shape_t({3, 4}))
                                 .template cast<TypeParam>());

        // No buffer provided, keepdims
       {ndarray<TypeParam> y = std(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y(shape_t({1, 3, 4}));
        ndarray<TypeParam> z = std(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncComplex, TestStdComplex) {
        ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());
        x += (arange<int>(0, 5 * 3 * 4, 1)
              .reshape(shape_t({5, 3, 4}))
              .template cast<TypeParam>()) * j<TypeParam>();
        ndarray<TypeParam> gt = (r_<int>({24, })
                                 .broadcast_to(shape_t({3, 4}))
                                 .template cast<TypeParam>());

        // No buffer provided, keepdims
       {ndarray<TypeParam> y = std(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y(shape_t({1, 3, 4}));
        ndarray<TypeParam> z = std(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncFloatComplex, TestMean) {
        ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());
        ndarray gt = (arange<int>(24, 36, 1)
                      .reshape(shape_t({3, 4}))
                      .template cast<TypeParam>());

        // No buffer provided, keepdims
       {ndarray<TypeParam> y = mean(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y(shape_t({1, 3, 4}));
        ndarray<TypeParam> z = mean(x, 0, true, &y);
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
                  .reshape(shape_t({5, 3, 4})));
        auto gt = (r_<size_t>({                    4u,                    8u,                    2u,                    3u,
                                                   1u,                    3u, 18446744073709551596u,                   19u,
                                                  17u,                    7u,                    1u,                    9u})
                   .reshape(shape_t({3, 4})));

        // No buffer provided, keepdims
       {auto y = min(x, 0);
        ASSERT_TRUE(all((y == gt).ravel(), 0)[{0}]);};

        // Buffer provided, dropdims
       {ndarray<size_t> y(shape_t({1, 3, 4}));
        auto z = min(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(all((y == gt).ravel(), 0)[{0}]);};
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
                  .reshape(shape_t({5, 3, 4}))
                  .template cast<TypeParam>());
        auto gt = (r_<int>({-17, -23, -21, -23,
                            -17, -25, -20, -20,
                             -8, -22,  -9, -15})
                   .reshape(shape_t({3, 4}))
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {ndarray<TypeParam> y = min(x, 0);
        ASSERT_TRUE(all((y == gt).ravel(), 0)[{0}]);};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y(shape_t({1, 3, 4}));
        ndarray<TypeParam> z = min(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(all((y == gt).ravel(), 0)[{0}]);};
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
                  .reshape(shape_t({5, 3, 4}))
                  .template cast<TypeParam>());
        auto gt = (r_<double>({-1.73222065, -3.25025675, -1.1708768 , -1.03443742,
                               -2.3840612 ,  0.24594309, -2.59267573, -1.52151508,
                               -1.52435019, -1.8137003 , -0.80477885, -1.10491782})
                   .reshape(shape_t({3, 4}))
                   .template cast<TypeParam>());

        // No buffer provided, keepdims
       {ndarray<TypeParam> y = min(x, 0);
        ASSERT_TRUE(allclose(y, gt));};

        // Buffer provided, dropdims
       {ndarray<TypeParam> y(shape_t({1, 3, 4}));
        ndarray<TypeParam> z = min(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestCeil) {
        ndarray<TypeParam> x = (arange<TypeParam>(-3.1, 3.9, 0.5)
                                .reshape(shape_t({2, 7})));
        ndarray<TypeParam> truth = (r_(std::vector<TypeParam> {-3, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 4})
                                    .reshape(shape_t({2, 7})));

        // No buffer provided
       {ndarray<TypeParam> y = ceil<TypeParam>(x);
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = truth.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> _y = ceil<TypeParam>(x, &y);
        ASSERT_TRUE(_y.equals(y));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = truth.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestFloor) {
        ndarray<TypeParam> x = (arange<TypeParam>(-3.1, 3.9, 0.5)
                                .reshape(shape_t({2, 7})));
        ndarray<TypeParam> truth = (r_(std::vector<TypeParam> {-4, -3, -3, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 3})
                                    .reshape(shape_t({2, 7})));

        // No buffer provided
       {ndarray<TypeParam> y = floor<TypeParam>(x);
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = truth.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> _y = floor<TypeParam>(x, &y);
        ASSERT_TRUE(_y.equals(y));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = truth.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}
    }

    TYPED_TEST_P(TestNdFuncIntFloat, TestClip) {
        ndarray<TypeParam> x = (arange<int>(3, 9, 1)
                                .reshape(shape_t({2, 3}))
                                .template cast<TypeParam>());
        TypeParam const down = static_cast<TypeParam>(5);
        TypeParam const up = static_cast<TypeParam>(7);
        ndarray<TypeParam> truth = (r_(std::vector<TypeParam> {5, 5, 5, 6, 7, 7})
                                    .reshape(shape_t({2, 3})));

        // No buffer provided
       {ndarray<TypeParam> y = clip<TypeParam>(x, down, up);
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = truth.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> _y = clip<TypeParam>(x, down, up, &y);
        ASSERT_TRUE(_y.equals(y));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = truth.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}
    }

    TYPED_TEST_P(TestNdFuncSignedIntFloat, TestSign) {
        ndarray<TypeParam> x = (arange<TypeParam>(-3, 3, 1)
                                .reshape(shape_t({2, 3})));
        ndarray<TypeParam> truth = (r_(std::vector<TypeParam> {-1, -1, -1, 0, 1, 1})
                                    .reshape(shape_t({2, 3})));

        // No buffer provided
       {ndarray<TypeParam> y = sign<TypeParam>(x);
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = truth.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> _y = sign<TypeParam>(x, &y);
        ASSERT_TRUE(_y.equals(y));
        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& correct = truth.data()[i];
            TypeParam const& tested = y.data()[i];
            ASSERT_EQ(correct, tested);
        }}
    }



    /* Proximity Tests ===================================================== */
    TYPED_TEST_P(TestNdFuncFloatComplex, TestIsClose) {
        // No buffer provided
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        ndarray<bool> y = isclose(x, x);
        for(auto it_y = y.begin(); it_y != y.end(); ++it_y) {
            ASSERT_TRUE(*it_y);
        }}

        // Buffer provided
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        ndarray<bool> y(x.shape());
        ndarray<bool> z = isclose(x, x, &y);
        ASSERT_TRUE(y.equals(z));

        for(auto it_y = y.begin(); it_y != y.end(); ++it_y) {
            ASSERT_TRUE(*it_y);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloatComplex, TestAllClose) {
        ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        bool y = allclose(x, x);
        ASSERT_TRUE(y);
    }



    /* Reduction Functions ================================================= */
    TYPED_TEST_P(TestNdFuncIntFloatComplex, TestSum) {
        // No buffer provided, keepdims
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = sum(x, 0);
        ASSERT_EQ(y.shape(), shape_t({3, 4}));
        TypeParam correct_elem = static_cast<TypeParam>(120.0);
        for(auto it_y = y.begin(); it_y != y.end();
            ++it_y, correct_elem += static_cast<TypeParam>(5.0)) {
            ASSERT_EQ(*it_y, correct_elem);
        }}

        // Buffer provided, dropdims
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y(shape_t({1, 3, 4}));
        ndarray<TypeParam> z = sum(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));

        TypeParam correct_elem = static_cast<TypeParam>(120.0);
        for(auto it_y = y.begin(); it_y != y.end();
            ++it_y, correct_elem += static_cast<TypeParam>(5.0)) {
            ASSERT_EQ(*it_y, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdFuncIntFloatComplex, TestProd) {
        // No buffer provided, keepdims
       {ndarray<TypeParam> x = (arange<int>(0, 2 * 3 * 4, 1)
                                .reshape(shape_t({2, 3, 4}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = prod(x, 0);
        ASSERT_EQ(y.shape(), shape_t({3, 4}));
        ASSERT_EQ(y.data()[0],  static_cast<TypeParam>(0.0));
        ASSERT_EQ(y.data()[1],  static_cast<TypeParam>(13.0));
        ASSERT_EQ(y.data()[2],  static_cast<TypeParam>(28.0));
        ASSERT_EQ(y.data()[3],  static_cast<TypeParam>(45.0));
        ASSERT_EQ(y.data()[4],  static_cast<TypeParam>(64.0));
        ASSERT_EQ(y.data()[5],  static_cast<TypeParam>(85.0));
        ASSERT_EQ(y.data()[6],  static_cast<TypeParam>(108.0));
        ASSERT_EQ(y.data()[7],  static_cast<TypeParam>(133.0));
        ASSERT_EQ(y.data()[8],  static_cast<TypeParam>(160.0));
        ASSERT_EQ(y.data()[9],  static_cast<TypeParam>(189.0));
        ASSERT_EQ(y.data()[10], static_cast<TypeParam>(220.0));
        ASSERT_EQ(y.data()[11], static_cast<TypeParam>(253.0));}

        // Buffer provided, dropdims
       {ndarray<TypeParam> x = (arange<int>(0, 2 * 3 * 4, 1)
                                .reshape(shape_t({2, 3, 4}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y(shape_t({1, 3, 4}));
        ndarray<TypeParam> z = prod(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_EQ(y.data()[0],  static_cast<TypeParam>(0.0));
        ASSERT_EQ(y.data()[1],  static_cast<TypeParam>(13.0));
        ASSERT_EQ(y.data()[2],  static_cast<TypeParam>(28.0));
        ASSERT_EQ(y.data()[3],  static_cast<TypeParam>(45.0));
        ASSERT_EQ(y.data()[4],  static_cast<TypeParam>(64.0));
        ASSERT_EQ(y.data()[5],  static_cast<TypeParam>(85.0));
        ASSERT_EQ(y.data()[6],  static_cast<TypeParam>(108.0));
        ASSERT_EQ(y.data()[7],  static_cast<TypeParam>(133.0));
        ASSERT_EQ(y.data()[8],  static_cast<TypeParam>(160.0));
        ASSERT_EQ(y.data()[9],  static_cast<TypeParam>(189.0));
        ASSERT_EQ(y.data()[10], static_cast<TypeParam>(220.0));
        ASSERT_EQ(y.data()[11], static_cast<TypeParam>(253.0));}
    }

    TEST(TestNdFunc, TestAny) {
        // No buffer provided, keepdims
       {ndarray<bool> x = zeros<bool>({2, 3});
        x[{1, 1}] = true;

        ndarray<bool> y = any(x, 0);
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.data()[0],  false);
        ASSERT_EQ(y.data()[1],  true);
        ASSERT_EQ(y.data()[2],  false);}

       {ndarray<bool> x = zeros<bool>({2, 3});
        x[{1, 1}] = true;

        ndarray<bool> y(shape_t({1, 3}));
        ndarray<bool> z = any(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_EQ(y.shape(), shape_t({1, 3}));
        ASSERT_EQ(y.data()[0],  false);
        ASSERT_EQ(y.data()[1],  true);
        ASSERT_EQ(y.data()[2],  false);}
    }

    TEST(TestNdFunc, TestAll) {
        // No buffer provided, keepdims
       {ndarray<bool> x = zeros<bool>({2, 3});
        x[{1, 1}] = true;

        ndarray<bool> y = all(x, 0);
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.data()[0],  false);
        ASSERT_EQ(y.data()[1],  false);
        ASSERT_EQ(y.data()[2],  false);}

       {ndarray<bool> x = zeros<bool>({2, 3});
        x[{1, 1}] = true;

        ndarray<bool> y(shape_t({1, 3}));
        ndarray<bool> z = all(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_EQ(y.shape(), shape_t({1, 3}));
        ASSERT_EQ(y.data()[0],  false);
        ASSERT_EQ(y.data()[1],  false);
        ASSERT_EQ(y.data()[2],  false);}
    }



    /* Trigonometric Functions ============================================= */
    TYPED_TEST_P(TestNdFuncFloat, TestSin) {
        // No buffer provided
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = sin(x);
        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::sin(*it_x));
        }}

        // Buffer provided
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
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
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = cos(x);
        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::cos(*it_x));
        }}

        // Buffer provided
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
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
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = tan(x);
        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::tan(*it_x));
        }}

        // Buffer provided
       {ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
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
        ndarray<TypeParam> x = (r_<TypeParam>({0, 1/M_SQRT2, 1, 1/M_SQRT2, 0, -1/M_SQRT2, -1, -1/M_SQRT2})
                                .reshape(shape_t({2, 4})));
        ndarray<TypeParam> gt = (r_<TypeParam>({0, M_PI/4, M_PI/2, M_PI/4, 0, -M_PI/4, -M_PI/2, -M_PI/4})
                                 .reshape(shape_t({2, 4})));

        // No buffer provided
       {ndarray<TypeParam> y = arcsin(x);
        ASSERT_TRUE(allclose(y, gt));}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = arcsin(x, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}

        // Out-of-Range
       {ndarray<TypeParam> x = r_<TypeParam>({-1.1, 1.1});
        ndarray<TypeParam> y = arcsin(x);
        ASSERT_TRUE(std::isnan(y[{0}]));
        ASSERT_TRUE(std::isnan(y[{1}]));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestArcCos) {
        ndarray<TypeParam> x = (r_<TypeParam>({1, 1/M_SQRT2, 0, -1/M_SQRT2, -1, -1/M_SQRT2, 0, 1/M_SQRT2})
                                .reshape(shape_t({2, 4})));
        ndarray<TypeParam> gt = (r_<TypeParam>({0, M_PI/4, M_PI/2, 3*M_PI/4, M_PI, 3*M_PI/4, M_PI/2, M_PI/4})
                                 .reshape(shape_t({2, 4})));

        // No buffer provided
       {ndarray<TypeParam> y = arccos(x);
        ASSERT_TRUE(allclose(y, gt));}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = arccos(x, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}

        // Out-of-Range
       {ndarray<TypeParam> x = r_<TypeParam>({-1.1, 1.1});
        ndarray<TypeParam> y = arccos(x);
        ASSERT_TRUE(std::isnan(y[{0}]));
        ASSERT_TRUE(std::isnan(y[{1}]));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestArcTan) {
        TypeParam const inf = std::numeric_limits<TypeParam>::infinity();
        ndarray<TypeParam> x = (r_<TypeParam>({0, 1, inf, -1, 0, 1, -inf, -1})
                                .reshape(shape_t({2, 4})));
        ndarray<TypeParam> gt = (r_<TypeParam>({0, M_PI/4, M_PI/2, -M_PI/4, 0, M_PI/4, -M_PI/2, -M_PI/4})
                                 .reshape(shape_t({2, 4})));

        // No buffer provided
       {ndarray<TypeParam> y = arctan(x);
        ASSERT_TRUE(allclose(y, gt));}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = arctan(x, &y);
        ASSERT_TRUE(y.equals(z));
        ASSERT_TRUE(allclose(y, gt));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestArcTan2) {
        ndarray<TypeParam> x = (r_<TypeParam>({1, 1/M_SQRT2, 0, -1/M_SQRT2, -1, -1/M_SQRT2, 0, 1/M_SQRT2})
                                .reshape(shape_t({2, 4})));
        ndarray<TypeParam> y = (r_<TypeParam>({0, 1/M_SQRT2, 1, 1/M_SQRT2, 0, -1/M_SQRT2, -1, -1/M_SQRT2})
                                .reshape(shape_t({2, 4})));
        ndarray<TypeParam> gt = (r_<TypeParam>({0, M_PI/4, M_PI/2, 3*M_PI/4, M_PI, -3*M_PI/4, -M_PI/2, -M_PI/4})
                                 .reshape(shape_t({2, 4})));

        // No buffer provided
       {ndarray<TypeParam> res = arctan2(y, x);
        ASSERT_TRUE(allclose(res, gt));}

        // Buffer provided
       {ndarray<TypeParam> res(shape_t({2, 4}));
        ndarray<TypeParam> z = arctan2(y, x, &res);
        ASSERT_TRUE(res.equals(z));
        ASSERT_TRUE(allclose(res, gt));}

        // Broadcasting
       {TypeParam const sqrt_3 = std::sqrt(3.0);
        ndarray<TypeParam> x = r_<TypeParam>({1, });
        ndarray<TypeParam> y = r_<TypeParam>({0, 1/sqrt_3, 1, -1/sqrt_3, -1});
        ndarray<TypeParam> gt = r_<TypeParam>({0, M_PI/6, M_PI/4, -M_PI/6, -M_PI/4});

        ndarray<TypeParam> res = arctan2(y, x);
        ASSERT_TRUE(allclose(res, gt));}
    }



    /* Unit Transformation ================================================= */
    TYPED_TEST_P(TestNdFuncFloat, TestDeg2Rad) {
        ndarray<TypeParam> ratio = r_(std::vector<TypeParam>({0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75}));
        ndarray<TypeParam> deg = ratio * ndarray<TypeParam>(180.0);
        ndarray<TypeParam> rad_gt = ratio * ndarray<TypeParam>(M_PI);

        // No buffer provided
       {ndarray<TypeParam> rad = deg2rad<TypeParam>(deg);
        ASSERT_TRUE(allclose(rad, rad_gt));}

        // Buffer provided
       {ndarray<TypeParam> rad(deg.shape());
        ndarray<TypeParam> _rad = deg2rad<TypeParam>(deg, &rad);
        ASSERT_TRUE(_rad.equals(rad));
        ASSERT_TRUE(allclose(rad, rad_gt));}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestRad2Deg) {
        ndarray<TypeParam> ratio = r_(std::vector<TypeParam>({0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75}));
        ndarray<TypeParam> rad = ratio * ndarray<TypeParam>(M_PI);
        ndarray<TypeParam> deg_gt = ratio * ndarray<TypeParam>(180.0);

        // No buffer provided
       {ndarray<TypeParam> deg = rad2deg<TypeParam>(rad);
        ASSERT_TRUE(allclose(deg, deg_gt));}

        // Buffer provided
       {ndarray<TypeParam> deg(rad.shape());
        ndarray<TypeParam> _deg = rad2deg<TypeParam>(rad, &deg);
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
       {ndarray<TypeParam> y = sinc(x);
        ASSERT_TRUE(allclose(y, gt, 1e-5, 1e-7));}

        // Buffer provided
       {ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = sinc(x, &y);
        ASSERT_TRUE(z.equals(y));
        ASSERT_TRUE(allclose(y, gt, 1e-5, 1e-7));}
    }



    /* Complex Numbers ===================================================== */
    TYPED_TEST_P(TestNdFuncFloat, TestReal) {
        // Contiguous array
       {ndarray<std::complex<TypeParam>> x(shape_t({5, 3, 4}));
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                 -static_cast<TypeParam>(i + 1.0));
        }

        ndarray<TypeParam> y = real(x);
        ASSERT_EQ(y.shape(), x.shape());

        auto it_y = y.begin();
        for(auto it_x = x.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).real(), *it_y);
        }}

        // Non-contiguous array
       {ndarray<std::complex<TypeParam>> _x(shape_t({5, 3, 8}));
        for(size_t i = 0; i < _x.size(); ++i) {
            _x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                  -static_cast<TypeParam>(i + 1.0));
        }
        auto x = _x({util::slice(), util::slice(), util::slice(0, 8, 2)});

        ndarray<TypeParam> y = real(x);
        ASSERT_EQ(y.shape(), x.shape());

        auto it_y = y.begin();
        for(auto it_x = x.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).real(), *it_y);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestImag) {
        // Contiguous array
       {ndarray<std::complex<TypeParam>> x(shape_t({5, 3, 4}));
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                 -static_cast<TypeParam>(i + 1.0));
        }

        ndarray<TypeParam> y = imag(x);
        ASSERT_EQ(y.shape(), x.shape());

        auto it_y = y.begin();
        for(auto it_x = x.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).imag(), *it_y);
        }}

        // Non-contiguous array
       {ndarray<std::complex<TypeParam>> _x(shape_t({5, 3, 8}));
        for(size_t i = 0; i < _x.size(); ++i) {
            _x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                  -static_cast<TypeParam>(i + 1.0));
        }
        auto x = _x({util::slice(), util::slice(), util::slice(0, 8, 2)});

        ndarray<TypeParam> y = imag(x);
        ASSERT_EQ(y.shape(), x.shape());

        auto it_y = y.begin();
        for(auto it_x = x.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).imag(), *it_y);
        }}
    }

    TYPED_TEST_P(TestNdFuncFloat, TestConj) {
        // No buffer provided
       {ndarray<std::complex<TypeParam>> x(shape_t({5, 3, 4}));
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                 -static_cast<TypeParam>(i + 1.0));
        }

        ndarray<std::complex<TypeParam>> y = conj(x);
        ASSERT_EQ(y.shape(), x.shape());

        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).real(), (*it_y).real());
            ASSERT_EQ((*it_x).imag(), -(*it_y).imag());
        }}

        // Buffer provided
       {ndarray<std::complex<TypeParam>> x(shape_t({5, 3, 4}));
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = std::complex<TypeParam>(static_cast<TypeParam>(i),
                                                 -static_cast<TypeParam>(i + 1.0));
        }

        ndarray<std::complex<TypeParam>> buffer(x.shape());
        ndarray<std::complex<TypeParam>> y = conj(x, &buffer);
        ASSERT_TRUE(buffer.equals(y));

        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ((*it_x).real(), (*it_y).real());
            ASSERT_EQ((*it_x).imag(), -(*it_y).imag());
        }}
    }



    /* Initialize tests ====================================================
     * Some test functions are not templated. We nevertheless list all of them here for
     * completeness.
     */
    // REGISTER_TYPED_TEST_CASE_P(TestNdFuncBool,
    //                            TestAny,
    //                            TestAll);
    // INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncBool, MyBoolTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncInt,
                               TestMinInt);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncInt, MyIntTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncSignedInt,
                               TestMinSignedInt);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncSignedInt, MySignedIntTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncFloat,
                               TestPi,
                               TestEuler,
                               TestLinSpace,
                               TestLog,
                               TestMinFloat,
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

    // REGISTER_TYPED_TEST_CASE_P(TestNdFuncBoolInt,);
    // INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncBoolInt, MyBoolIntTypes);

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
                               TestMean,
                               TestIsClose,
                               TestAllClose);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncFloatComplex, MyFloatComplexTypes);

    // REGISTER_TYPED_TEST_CASE_P(TestNdFuncSignedArithmetic,);
    // INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncSignedArithmetic, MySignedArithmeticTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdFuncArithmetic,
                               TestRUnderscore,
                               TestZeros,
                               TestOnes,
                               TestFull,
                               TestEye);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncArithmetic, MyArithmeticTypes);
}

#endif // TEST_NDFUNC_CPP
