// ############################################################################
// test_ndarray_operator.cpp
// =========================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_OPERATOR_CPP
#define TEST_NDARRAY_OPERATOR_CPP

#include <complex>

#include <gtest/gtest.h>
#include "test_type.hpp"

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T> class TestNdOperatorBool : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdOperatorBool);

    template <typename T> class TestNdOperatorInt : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdOperatorInt);

    template <typename T> class TestNdOperatorBoolInt : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdOperatorBoolInt);

    template <typename T> class TestNdOperatorIntFloat : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdOperatorIntFloat);

    template <typename T> class TestNdOperatorIntFloatComplex : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdOperatorIntFloatComplex);

    template <typename T> class TestNdOperatorArithmetic : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdOperatorArithmetic);



    TYPED_TEST_P(TestNdOperatorArithmetic, TestEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = zeros<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        lhs = rhs;
        for(auto it_lhs = lhs.begin(), it_rhs = rhs.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, *it_rhs);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = zeros<TypeParam>({20, 30, 40});
        TypeParam const rhs = static_cast<TypeParam>(3.0);

        lhs = rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = zeros<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        lhs = rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(auto it_lhs = lhs.begin(), it_rhs = rhs_bcast.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, *it_rhs);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloatComplex, TestPlusEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = ones<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        lhs += rhs;
        for(auto it_lhs = lhs.begin(), it_rhs = rhs.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, *it_rhs + static_cast<TypeParam>(1.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = zeros<TypeParam>({20, 30, 40});
        TypeParam const rhs = static_cast<TypeParam>(3.0);

        lhs += rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = zeros<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        lhs += rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(auto it_lhs = lhs.begin(), it_rhs = rhs_bcast.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, *it_rhs);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloatComplex, TestMinusEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = ones<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        lhs -= rhs;
        for(auto it_lhs = lhs.begin(), it_rhs = rhs.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, static_cast<TypeParam>(1.0) - (*it_rhs) );
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = zeros<TypeParam>({20, 30, 40});
        TypeParam const rhs = static_cast<TypeParam>(3.0);

        lhs -= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], -rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = zeros<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        lhs -= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(auto it_lhs = lhs.begin(), it_rhs = rhs_bcast.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, - (*it_rhs));
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloatComplex, TestTimesEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = ones<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        lhs *= rhs;
        for(auto it_lhs = lhs.begin(), it_rhs = rhs.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, *it_rhs);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = ones<TypeParam>({20, 30, 40});
        TypeParam const rhs = static_cast<TypeParam>(3.0);

        lhs *= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = ones<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        lhs *= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(auto it_lhs = lhs.begin(), it_rhs = rhs_bcast.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, *it_rhs);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloatComplex, TestDivideEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 2.0);

        lhs /= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = ((arange<int>(0, 20 * 30 * 40, 1) * 3)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(3.0);

        lhs /= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = ((arange<int>(0, 20 * 30 * 40, 1) * 4)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 4.0);

        lhs /= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i));
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestModEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs %= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) % static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = ((arange<int>(0, 20 * 30 * 40, 1) * 7)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs %= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(0.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs %= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) % static_cast<TypeParam>(7.0));
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestAndEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs &= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) & static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs &= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) & static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs &= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) & static_cast<TypeParam>(7.0));
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestOrEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs |= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) | static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs |= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) | static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs |= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) | static_cast<TypeParam>(7.0));
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestXorEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs ^= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) ^ static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs ^= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) ^ static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs ^= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) ^ static_cast<TypeParam>(7.0));
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestShiftLeftEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs <<= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) << static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs <<= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) << static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs <<= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) << static_cast<TypeParam>(7.0));
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestShiftRightEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs >>= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) >> static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs >>= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) >> static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs >>= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) >> static_cast<TypeParam>(7.0));
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestPlusPlus) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape({20, 30, 40})
                                .template cast<TypeParam>());
        ndarray<TypeParam> y = x.copy();

        ++x;
        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, static_cast<TypeParam>(1) + (*it_y));
        }
    }

    TYPED_TEST_P(TestNdOperatorInt, TestMinusMinus) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape({20, 30, 40})
                                .template cast<TypeParam>());
        ndarray<TypeParam> y = x.copy();

        --x;
        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(static_cast<TypeParam>(1) + (*it_x), *it_y);
        }
    }

    TYPED_TEST_P(TestNdOperatorIntFloatComplex, TestUnaryMinus) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape({20, 30, 40})
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = -x;

        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, - (*it_x));
        }
    }

    TYPED_TEST_P(TestNdOperatorInt, TestUnaryBitFlip) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape({20, 30, 40})
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = ~x;

        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, ~ (*it_x));
        }
    }

    TYPED_TEST_P(TestNdOperatorBool, TestNot) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape({20, 30, 40})
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = !x;

        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, !(*it_x));
        }
    }

    TYPED_TEST_P(TestNdOperatorIntFloatComplex, TestPlus) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs + rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] + rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs + rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] + rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs + rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) + (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloatComplex, TestMinus) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs - rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] - rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs - rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] - rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs - rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) - (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloatComplex, TestTimes) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs * rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] * rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs * rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] * rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs * rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) * (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloatComplex, TestDivide) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2 + 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs / rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] / rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs / rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] / rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 1 * 40, 1) + 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs / rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) / (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestMod) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2 + 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs % rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] % rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs % rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] % rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 1 * 40, 1) + 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs % rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) % (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestAnd) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs & rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] & rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs & rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] & rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs & rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) & (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestOr) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs | rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] | rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs | rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] | rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs | rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) | (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestXor) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs ^ rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] ^ rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs ^ rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] ^ rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs ^ rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) ^ (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestShiftLeft) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs << rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] << rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs << rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] << rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs << rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) << (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorInt, TestShiftRight) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * 2)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs >> rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] >> rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs >> rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] >> rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs >> rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) >> (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorBool, TestAndAnd) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs && rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] && rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs && rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] && rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs && rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) && (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorBool, TestOrOr) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs || rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] || rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<TypeParam> out = lhs || rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] || rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<TypeParam> out = lhs || rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            TypeParam const& tested_elem = *it_out;
            TypeParam const& correct_elem = (*it_lhs) || (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorBoolInt, TestEqualEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs == rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] == rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<bool> out = lhs == rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] == rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<bool> out = lhs == rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            bool const& tested_elem = *it_out;
            bool const& correct_elem = (*it_lhs) == (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorBoolInt, TestNotEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs != rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] != rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<bool> out = lhs != rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] != rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<bool> out = lhs != rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            bool const& tested_elem = *it_out;
            bool const& correct_elem = (*it_lhs) != (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloat, TestLess) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) + 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs < rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] < rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<bool> out = lhs < rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] < rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<bool> out = lhs < rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            bool const& tested_elem = *it_out;
            bool const& correct_elem = (*it_lhs) < (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloat, TestLessEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) + 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs <= rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] <= rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<bool> out = lhs <= rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] <= rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<bool> out = lhs <= rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            bool const& tested_elem = *it_out;
            bool const& correct_elem = (*it_lhs) <= (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloat, TestGreater) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) + 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs > rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] > rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<bool> out = lhs > rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] > rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<bool> out = lhs > rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            bool const& tested_elem = *it_out;
            bool const& correct_elem = (*it_lhs) > (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdOperatorIntFloat, TestGreaterEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) + 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs >= rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] >= rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape({20, 30, 40})
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        ndarray<bool> out = lhs >= rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] >= rhs);
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 1, 1)
                                  .reshape({20, 30, 1})
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape({20, 1, 40})
                                  .template cast<TypeParam>());

        shape_t const shape_out({20, 30, 40});
        ndarray<bool> out = lhs >= rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_out);
        ndarray<TypeParam> lhs_bcast = lhs.broadcast_to(shape_out);
        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(shape_out);
        for(auto it_lhs = lhs_bcast.begin(), it_rhs = rhs_bcast.begin(), it_out = out.begin();
            it_out != out.end();
            ++it_lhs, ++it_rhs, ++it_out) {
            bool const& tested_elem = *it_out;
            bool const& correct_elem = (*it_lhs) >= (*it_rhs);
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }



    // Initialize tests ====================================================
    REGISTER_TYPED_TEST_CASE_P(TestNdOperatorBool,
                               TestNot,
                               TestAndAnd,
                               TestOrOr);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdOperatorBool, MyBoolTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdOperatorInt,
                               TestModEqual,
                               TestAndEqual,
                               TestOrEqual,
                               TestXorEqual,
                               TestShiftLeftEqual,
                               TestShiftRightEqual,
                               TestPlusPlus,
                               TestMinusMinus,
                               TestUnaryBitFlip,
                               TestMod,
                               TestAnd,
                               TestOr,
                               TestXor,
                               TestShiftLeft,
                               TestShiftRight);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdOperatorInt, MyIntTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdOperatorBoolInt,
                               TestEqualEqual,
                               TestNotEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdOperatorBoolInt, MyBoolIntTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdOperatorIntFloat,
                               TestLess,
                               TestLessEqual,
                               TestGreater,
                               TestGreaterEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdOperatorIntFloat, MyIntFloatTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdOperatorIntFloatComplex,
                               TestPlusEqual,
                               TestMinusEqual,
                               TestTimesEqual,
                               TestDivideEqual,
                               TestUnaryMinus,
                               TestPlus,
                               TestMinus,
                               TestTimes,
                               TestDivide);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdOperatorIntFloatComplex, MyIntFloatComplexTypes);

    REGISTER_TYPED_TEST_CASE_P(TestNdOperatorArithmetic,
                               TestEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdOperatorArithmetic, MyArithmeticTypes);
}

#endif // TEST_NDARRAY_OPERATOR_CPP
