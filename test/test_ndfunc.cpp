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
#include <vector>
#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    /* Constants =========================================================== */
    template <typename T>
    class TestNdFuncPi : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncPi);
    TYPED_TEST_P(TestNdFuncPi, TestPi) {
        TypeParam const tested = pi<TypeParam>();
        TypeParam const correct = static_cast<TypeParam>(M_PI);

        ASSERT_EQ(tested, correct);
    }
    typedef ::testing::Types<float, double> MyNdFuncPiTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncPi,
                               TestPi);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncPi, MyNdFuncPiTypes);

    template <typename T>
    class TestNdFuncEuler : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncEuler);
    TYPED_TEST_P(TestNdFuncEuler, TestEuler) {
        TypeParam const tested = e<TypeParam>();
        TypeParam const correct = static_cast<TypeParam>(M_E);

        ASSERT_EQ(tested, correct);
    }
    typedef ::testing::Types<float, double> MyNdFuncEulerTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncEuler,
                               TestEuler);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncEuler, MyNdFuncEulerTypes);

    /* Utility ============================================================= */
    template <typename T>
    class TestNdFuncUtility : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncUtility);
    TYPED_TEST_P(TestNdFuncUtility, TestRUnderscore) {
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

    TYPED_TEST_P(TestNdFuncUtility, TestZeros) {
        ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, static_cast<TypeParam>(0.0));
        }
    }

    TYPED_TEST_P(TestNdFuncUtility, TestOnes) {
        ndarray<TypeParam> x = ones<TypeParam>({5, 3, 4});
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, static_cast<TypeParam>(1.0));
        }
    }

    TYPED_TEST_P(TestNdFuncUtility, TestFull) {
        TypeParam value = static_cast<TypeParam>(3.0);
        ndarray<TypeParam> x = full<TypeParam>({5, 3, 4}, value);
        for(auto it = x.begin(); it != x.end(); ++it) {
            ASSERT_EQ(*it, value);
        }
    }

    TYPED_TEST_P(TestNdFuncUtility, TestEye) {
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

    typedef ::testing::Types<bool, int, size_t,
                             float, double,
                             std::complex<float>,
                             std::complex<double>> MyNdFuncUtilityTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncUtility,
                               TestRUnderscore,
                               TestZeros,
                               TestOnes,
                               TestFull,
                               TestEye);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncUtility, MyNdFuncUtilityTypes);

    template <typename T>
    class TestNdFuncArange : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncArange);
    TYPED_TEST_P(TestNdFuncArange, TestArange) {
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
    typedef ::testing::Types<int, float, double> MyNdFuncArangeTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncArange,
                               TestArange);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncArange, MyNdFuncArangeTypes);

    template <typename T>
    class TestNdFuncLinSpace : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncLinSpace);
    TYPED_TEST_P(TestNdFuncLinSpace, TestLinSpace) {
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
    typedef ::testing::Types<float, double> MyNdFuncLinSpaceTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncLinSpace,
                               TestLinSpace);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncLinSpace, MyNdFuncLinSpaceTypes);

    /* Math Functions ====================================================== */
    template <typename T>
    class TestNdFuncAbs : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncAbs);
    TYPED_TEST_P(TestNdFuncAbs, TestAbs) {
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
    typedef ::testing::Types<int,
                             float, double,
                             std::complex<float>,
                             std::complex<double>> MyNdFuncAbsTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncAbs,
                               TestAbs);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncAbs, MyNdFuncAbsTypes);

    template <typename T>
    class TestNdFuncIsClose : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncIsClose);
    TYPED_TEST_P(TestNdFuncIsClose, TestIsClose) {
        // No buffer provided
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<bool> y = isclose(x, x);
        for(auto it_y = y.begin(); it_y != y.end(); ++it_y) {
            ASSERT_TRUE(*it_y);
        }}

        // Buffer provided
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<bool> y(x.shape());
        ndarray<bool> z = isclose(x, x, &y);
        ASSERT_TRUE(y.equals(z));

        for(auto it_y = y.begin(); it_y != y.end(); ++it_y) {
            ASSERT_TRUE(*it_y);
        }}
    }
    typedef ::testing::Types<float, double,
                             std::complex<float>,
                             std::complex<double>> MyNdFuncIsCloseTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncIsClose,
                               TestIsClose);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncIsClose, MyNdFuncIsCloseTypes);

    template <typename T>
    class TestNdFuncAllClose : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncAllClose);
    TYPED_TEST_P(TestNdFuncAllClose, TestAllClose) {
        ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        bool y = allclose(x, x);
        ASSERT_TRUE(y);
    }
    typedef ::testing::Types<float, double,
                             std::complex<float>,
                             std::complex<double>> MyNdFuncAllCloseTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncAllClose,
                               TestAllClose);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncAllClose, MyNdFuncAllCloseTypes);

    template <typename T>
    class TestNdFuncSum : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncSum);
    TYPED_TEST_P(TestNdFuncSum, TestSum) {
        // No buffer provided, keepdims
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y = sum(x, 0);
        ASSERT_EQ(y.shape(), shape_t({3, 4}));
        TypeParam correct_elem = static_cast<TypeParam>(120.0);
        for(auto it_y = y.begin(); it_y != y.end();
            ++it_y, correct_elem += static_cast<TypeParam>(5.0)) {
            ASSERT_EQ(*it_y, correct_elem);
        }}

        // Buffer provided, dropdims
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y(shape_t({1, 3, 4}));
        ndarray<TypeParam> z = sum(x, 0, true, &y);
        ASSERT_TRUE(y.equals(z));

        TypeParam correct_elem = static_cast<TypeParam>(120.0);
        for(auto it_y = y.begin(); it_y != y.end();
            ++it_y, correct_elem += static_cast<TypeParam>(5.0)) {
            ASSERT_EQ(*it_y, correct_elem);
        }}
    }
    typedef ::testing::Types<int, size_t,
                             float, double,
                             std::complex<float>,
                             std::complex<double>> MyNdFuncSumTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncSum,
                               TestSum);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncSum, MyNdFuncSumTypes);

    template <typename T>
    class TestNdFuncProd : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncProd);
    TYPED_TEST_P(TestNdFuncProd, TestProd) {
        // No buffer provided, keepdims
       {ndarray<TypeParam> x = zeros<TypeParam>({2, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

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
       {ndarray<TypeParam> x = zeros<TypeParam>({2, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

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
    typedef ::testing::Types<int, size_t,
                             float, double,
                             std::complex<float>,
                             std::complex<double>> MyNdFuncProdTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncProd,
                               TestProd);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncProd, MyNdFuncProdTypes);

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

    template <typename T>
    class TestNdFuncSin : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncSin);
    TYPED_TEST_P(TestNdFuncSin, TestSin) {
        // No buffer provided
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y = sin(x);
        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::sin(*it_x));
        }}

        // Buffer provided
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = sin(x, &y);
        ASSERT_TRUE(y.equals(z));

        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::sin(*it_x));
        }}
    }
    typedef ::testing::Types<float, double> MyNdFuncSinTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncSin,
                               TestSin);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncSin, MyNdFuncSinTypes);

    template <typename T>
    class TestNdFuncCos : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncCos);
    TYPED_TEST_P(TestNdFuncCos, TestCos) {
        // No buffer provided
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y = cos(x);
        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::cos(*it_x));
        }}

        // Buffer provided
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = cos(x, &y);
        ASSERT_TRUE(y.equals(z));

        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::cos(*it_x));
        }}
    }
    typedef ::testing::Types<float, double> MyNdFuncCosTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncCos,
                               TestCos);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncCos, MyNdFuncCosTypes);

    template <typename T>
    class TestNdFuncTan : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncTan);
    TYPED_TEST_P(TestNdFuncTan, TestTan) {
        // No buffer provided
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y = tan(x);
        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::tan(*it_x));
        }}

        // Buffer provided
       {ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y(x.shape());
        ndarray<TypeParam> z = tan(x, &y);
        ASSERT_TRUE(y.equals(z));

        for(auto it_x = x.begin(), it_y = y.begin();
            it_y != y.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, std::tan(*it_x));
        }}
    }
    typedef ::testing::Types<float, double> MyNdFuncTanTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncTan,
                               TestTan);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncTan, MyNdFuncTanTypes);

    template <typename T>
    class TestNdFuncDeg2Rad : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncDeg2Rad);
    TYPED_TEST_P(TestNdFuncDeg2Rad, TestDeg2Rad) {
        ndarray<TypeParam> ratio = r_(std::vector<TypeParam> {0, 0.25, 0.5, 0.75, 1.0, -0.25, -0.5, -0.75});
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
    typedef ::testing::Types<float, double> MyNdFuncDeg2RadTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncDeg2Rad,
                               TestDeg2Rad);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncDeg2Rad, MyNdFuncDeg2RadTypes);    template <typename T>
    class TestNdFuncReal : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncReal);

    TYPED_TEST_P(TestNdFuncReal, TestReal) {
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

    typedef ::testing::Types<float, double> MyNdFuncRealTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncReal,
                               TestReal);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncReal, MyNdFuncRealTypes);

    template <typename T>
    class TestNdFuncImag : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncImag);

    TYPED_TEST_P(TestNdFuncImag, TestImag) {
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

    typedef ::testing::Types<float, double> MyNdFuncImagTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncImag,
                               TestImag);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncImag, MyNdFuncImagTypes);

    template <typename T>
    class TestNdFuncConj : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdFuncConj);

    TYPED_TEST_P(TestNdFuncConj, TestConj) {
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

    typedef ::testing::Types<float, double> MyNdFuncConjTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdFuncConj,
                               TestConj);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdFuncConj, MyNdFuncConjTypes);
}

#endif // TEST_NDFUNC_CPP
