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
