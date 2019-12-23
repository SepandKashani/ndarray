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
    template <typename T>
    class TestNdArrayOperatorEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorEqual);
    TYPED_TEST_P(TestNdArrayOperatorEqual, TestEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = zeros<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        lhs = rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(auto it_lhs = lhs.begin(), it_rhs = rhs_bcast.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, *it_rhs);
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorEqual,
                               TestEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorEqual, MyArithmeticTypes);



    template <typename T>
    class TestNdArrayOperatorPlusEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorPlusEqual);
    TYPED_TEST_P(TestNdArrayOperatorPlusEqual, TestPlusEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = ones<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        lhs += rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(auto it_lhs = lhs.begin(), it_rhs = rhs_bcast.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, *it_rhs);
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorPlusEqual,
                               TestPlusEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorPlusEqual, MyIntFloatComplexTypes);



    template <typename T>
    class TestNdArrayOperatorMinusEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorMinusEqual);
    TYPED_TEST_P(TestNdArrayOperatorMinusEqual, TestMinusEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = ones<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        lhs -= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(auto it_lhs = lhs.begin(), it_rhs = rhs_bcast.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, - (*it_rhs));
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorMinusEqual,
                               TestMinusEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorMinusEqual, MyIntFloatComplexTypes);



    template <typename T>
    class TestNdArrayOperatorTimesEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorTimesEqual);
    TYPED_TEST_P(TestNdArrayOperatorTimesEqual, TestTimesEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = ones<TypeParam>({20, 30, 40});
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        lhs *= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(auto it_lhs = lhs.begin(), it_rhs = rhs_bcast.begin();
            it_lhs != lhs.end();
            ++it_lhs, ++it_rhs) {
            ASSERT_EQ(*it_lhs, *it_rhs);
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorTimesEqual,
                               TestTimesEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorTimesEqual, MyIntFloatComplexTypes);



    template <typename T>
    class TestNdArrayOperatorDivideEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorDivideEqual);
    TYPED_TEST_P(TestNdArrayOperatorDivideEqual, TestDivideEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(2))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 2.0);

        lhs /= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(3))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(3.0);

        lhs /= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(4))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 4.0);

        lhs /= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i));
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorDivideEqual,
                               TestDivideEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorDivideEqual, MyIntFloatComplexTypes);



    template <typename T>
    class TestNdArrayOperatorModEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorModEqual);
    TYPED_TEST_P(TestNdArrayOperatorModEqual, TestModEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs %= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) % static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(7))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs %= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(0.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs %= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) % static_cast<TypeParam>(7.0));
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorModEqual,
                               TestModEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorModEqual, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorAndEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorAndEqual);
    TYPED_TEST_P(TestNdArrayOperatorAndEqual, TestAndEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs &= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) & static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs &= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) & static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs &= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) & static_cast<TypeParam>(7.0));
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorAndEqual,
                               TestAndEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorAndEqual, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorOrEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorOrEqual);
    TYPED_TEST_P(TestNdArrayOperatorOrEqual, TestOrEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs |= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) | static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs |= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) | static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs |= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) | static_cast<TypeParam>(7.0));
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorOrEqual,
                               TestOrEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorOrEqual, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorXorEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorXorEqual);
    TYPED_TEST_P(TestNdArrayOperatorXorEqual, TestXorEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs ^= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) ^ static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs ^= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) ^ static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs ^= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) ^ static_cast<TypeParam>(7.0));
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorXorEqual,
                               TestXorEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorXorEqual, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorShiftLeftEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorShiftLeftEqual);
    TYPED_TEST_P(TestNdArrayOperatorShiftLeftEqual, TestShiftLeftEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs <<= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) << static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs <<= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) << static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs <<= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) << static_cast<TypeParam>(7.0));
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorShiftLeftEqual,
                               TestShiftLeftEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorShiftLeftEqual, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorShiftRightEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorShiftRightEqual);
    TYPED_TEST_P(TestNdArrayOperatorShiftRightEqual, TestShiftRightEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 30, 40}, 5.0);

        lhs >>= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) >> static_cast<TypeParam>(5.0));
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        TypeParam const rhs = static_cast<TypeParam>(7.0);

        lhs >>= rhs;
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) >> static_cast<TypeParam>(7.0));
        }}

        // RHS broadcasts
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = full<TypeParam>({20, 1, 40}, 7.0);

        lhs >>= rhs;

        ndarray<TypeParam> rhs_bcast = rhs.broadcast_to(lhs.shape());
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(lhs.data()[i], static_cast<TypeParam>(i) >> static_cast<TypeParam>(7.0));
        }}
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorShiftRightEqual,
                               TestShiftRightEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorShiftRightEqual, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorPlusPlus : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorPlusPlus);
    TYPED_TEST_P(TestNdArrayOperatorPlusPlus, TestPlusPlus) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape(shape_t({20, 30, 40}))
                                .template cast<TypeParam>());
        ndarray<TypeParam> y = x.copy();

        ++x;
        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, static_cast<TypeParam>(1) + (*it_y));
        }
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorPlusPlus,
                               TestPlusPlus);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorPlusPlus, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorMinusMinus : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorMinusMinus);
    TYPED_TEST_P(TestNdArrayOperatorMinusMinus, TestMinusMinus) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape(shape_t({20, 30, 40}))
                                .template cast<TypeParam>());
        ndarray<TypeParam> y = x.copy();

        --x;
        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(static_cast<TypeParam>(1) + (*it_x), *it_y);
        }
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorMinusMinus,
                               TestMinusMinus);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorMinusMinus, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorUnaryMinus : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorUnaryMinus);
    TYPED_TEST_P(TestNdArrayOperatorUnaryMinus, TestUnaryMinus) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape(shape_t({20, 30, 40}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = -x;

        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, - (*it_x));
        }
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorUnaryMinus,
                               TestUnaryMinus);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorUnaryMinus, MyIntFloatComplexTypes);



    template <typename T>
    class TestNdArrayOperatorUnaryBitFlip : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorUnaryBitFlip);
    TYPED_TEST_P(TestNdArrayOperatorUnaryBitFlip, TestUnaryBitFlip) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape(shape_t({20, 30, 40}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = ~x;

        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, ~ (*it_x));
        }
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorUnaryBitFlip,
                               TestUnaryBitFlip);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorUnaryBitFlip, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorNot : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorNot);
    TYPED_TEST_P(TestNdArrayOperatorNot, TestNot) {
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape(shape_t({20, 30, 40}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = !x;

        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            ASSERT_EQ(*it_y, !(*it_x));
        }
    }
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorNot,
                               TestNot);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorNot, MyBoolTypes);



    template <typename T>
    class TestNdArrayOperatorPlus : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorPlus);
    TYPED_TEST_P(TestNdArrayOperatorPlus, TestPlus) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(2))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs + rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] + rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorPlus,
                               TestPlus);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorPlus, MyIntFloatComplexTypes);



    template <typename T>
    class TestNdArrayOperatorMinus : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorMinus);
    TYPED_TEST_P(TestNdArrayOperatorMinus, TestMinus) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(2))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs - rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] - rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorMinus,
                               TestMinus);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorMinus, MyIntFloatComplexTypes);



    template <typename T>
    class TestNdArrayOperatorTimes : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorTimes);
    TYPED_TEST_P(TestNdArrayOperatorTimes, TestTimes) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(2))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs * rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] * rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorTimes,
                               TestTimes);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorTimes, MyIntFloatComplexTypes);



    template <typename T>
    class TestNdArrayOperatorDivide : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorDivide);
    TYPED_TEST_P(TestNdArrayOperatorDivide, TestDivide) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((ndarray<int>(2) * arange<int>(0, 20 * 30 * 40, 1) + ndarray<int>(1))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs / rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] / rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 1 * 40, 1) + ndarray<int>(1))
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorDivide,
                               TestDivide);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorDivide, MyIntFloatComplexTypes);



    template <typename T>
    class TestNdArrayOperatorMod : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorMod);
    TYPED_TEST_P(TestNdArrayOperatorMod, TestMod) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((ndarray<int>(2) * arange<int>(0, 20 * 30 * 40, 1) + ndarray<int>(1))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs % rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] % rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 1 * 40, 1) + ndarray<int>(1))
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorMod,
                               TestMod);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorMod, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorAnd : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorAnd);
    TYPED_TEST_P(TestNdArrayOperatorAnd, TestAnd) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(2))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs & rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] & rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorAnd,
                               TestAnd);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorAnd, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorOr : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorOr);
    TYPED_TEST_P(TestNdArrayOperatorOr, TestOr) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(2))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs | rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] | rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorOr,
                               TestOr);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorOr, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorXor : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorXor);
    TYPED_TEST_P(TestNdArrayOperatorXor, TestXor) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(2))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs ^ rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] ^ rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorXor,
                               TestXor);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorXor, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorShiftLeft : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorShiftLeft);
    TYPED_TEST_P(TestNdArrayOperatorShiftLeft, TestShiftLeft) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(2))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs << rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] << rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorShiftLeft,
                               TestShiftLeft);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorShiftLeft, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorShiftRight : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorShiftRight);
    TYPED_TEST_P(TestNdArrayOperatorShiftRight, TestShiftRight) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) * ndarray<int>(2))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs >> rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] >> rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorShiftRight,
                               TestShiftRight);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorShiftRight, MyIntTypes);



    template <typename T>
    class TestNdArrayOperatorAndAnd : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorAndAnd);
    TYPED_TEST_P(TestNdArrayOperatorAndAnd, TestAndAnd) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs && rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] && rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorAndAnd,
                               TestAndAnd);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorAndAnd, MyBoolTypes);



    template <typename T>
    class TestNdArrayOperatorOrOr : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorOrOr);
    TYPED_TEST_P(TestNdArrayOperatorOrOr, TestOrOr) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<TypeParam> out = lhs || rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] || rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorOrOr,
                               TestOrOr);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorOrOr, MyBoolTypes);



    template <typename T>
    class TestNdArrayOperatorEqualEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorEqualEqual);
    TYPED_TEST_P(TestNdArrayOperatorEqualEqual, TestEqualEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs == rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] == rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorEqualEqual,
                               TestEqualEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorEqualEqual, MyBoolIntTypes);



    template <typename T>
    class TestNdArrayOperatorNotEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorNotEqual);
    TYPED_TEST_P(TestNdArrayOperatorNotEqual, TestNotEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs != rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] != rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorNotEqual,
                               TestNotEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorNotEqual, MyBoolIntTypes);



    template <typename T>
    class TestNdArrayOperatorLess : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorLess);
    TYPED_TEST_P(TestNdArrayOperatorLess, TestLess) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) + ndarray<int>(1))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs < rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] < rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorLess,
                               TestLess);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorLess, MyIntFloatTypes);



    template <typename T>
    class TestNdArrayOperatorLessEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorLessEqual);
    TYPED_TEST_P(TestNdArrayOperatorLessEqual, TestLessEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) + ndarray<int>(1))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs <= rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] <= rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorLessEqual,
                               TestLessEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorLessEqual, MyIntFloatTypes);



    template <typename T>
    class TestNdArrayOperatorGreater : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorGreater);
    TYPED_TEST_P(TestNdArrayOperatorGreater, TestGreater) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) + ndarray<int>(1))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs > rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] > rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorGreater,
                               TestGreater);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorGreater, MyIntFloatTypes);



    template <typename T>
    class TestNdArrayOperatorGreaterEqual : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayOperatorGreaterEqual);
    TYPED_TEST_P(TestNdArrayOperatorGreaterEqual, TestGreaterEqual) {
        // shape(LHS) == shape(RHS)
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = ((arange<int>(0, 20 * 30 * 40, 1) + ndarray<int>(1))
                                  .reshape(shape_t({20, 30, 40}))
                                  .template cast<TypeParam>());

        ndarray<bool> out = lhs >= rhs;
        ASSERT_TRUE(out.is_contiguous());
        ASSERT_EQ(out.shape(), shape_t({20, 30, 40}));
        for(size_t i = 0; i < lhs.size(); ++i) {
            ASSERT_EQ(out.data()[i], lhs.data()[i] >= rhs.data()[i]);
        }}

        // shape(RHS) == 1
       {ndarray<TypeParam> lhs = (arange<int>(0, 20 * 30 * 40, 1)
                                  .reshape(shape_t({20, 30, 40}))
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
                                  .reshape(shape_t({20, 30, 1}))
                                  .template cast<TypeParam>());
        ndarray<TypeParam> rhs = (arange<int>(0, 20 * 1 * 40, 1)
                                  .reshape(shape_t({20, 1, 40}))
                                  .template cast<TypeParam>());

        shape_t const shape_out = shape_t({20, 30, 40});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayOperatorGreaterEqual,
                               TestGreaterEqual);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayOperatorGreaterEqual, MyIntFloatTypes);
}

#endif // TEST_NDARRAY_OPERATOR_CPP
