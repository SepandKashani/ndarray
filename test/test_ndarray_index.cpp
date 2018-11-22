// ############################################################################
// test_ndarray_index.cpp
// ======================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_INDEX_CPP
#define TEST_NDARRAY_INDEX_CPP

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T>
    class TestNdArrayIndex : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayIndex);

    TYPED_TEST_P(TestNdArrayIndex, TestOperatorSquareBracket) {
        // Positive strides
        ndarray<TypeParam> x({3, 1, 4, 5});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        for(size_t i = 0; i < x.shape()[0]; ++i) {
            for(size_t j = 0; j < x.shape()[1]; ++j) {
                for(size_t k = 0; k < x.shape()[2]; ++k) {
                    for(size_t l = 0; l < x.shape()[3]; ++l) {
                        TypeParam const tested_elem = x[{i, j, k, l}];
                        int const offset = ((x.strides()[0] * i) +
                                            (x.strides()[1] * j) +
                                            (x.strides()[2] * k) +
                                            (x.strides()[3] * l));
                        byte_t* correct_addr = reinterpret_cast<byte_t*>(x.data()) + offset;
                        TypeParam const correct_elem = reinterpret_cast<TypeParam*>(correct_addr)[0];
                        ASSERT_EQ(tested_elem, correct_elem);
                    }
                }
            }
        }

        /* Can update entries. */
        TypeParam& tested_elem = x[{2, 0, 3, 4}];
        TypeParam const correct_elem = 500;
        tested_elem = correct_elem;
        ASSERT_EQ(tested_elem, correct_elem);

        // Negative strides
        shape_t shape({5, 3});
        stride_t strides({-3*static_cast<int>(sizeof(TypeParam)), sizeof(TypeParam)});
        TypeParam* data = new TypeParam[5 * 3];
        TypeParam* data_end = data + 15 - 1;
        for(size_t i = 0; i < 15; ++i) {
            data[i] = static_cast<TypeParam>(i);
        }
        ndarray<TypeParam> y(reinterpret_cast<byte_t*>(data_end), shape, strides);
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                TypeParam const tested_elem = y[{i, j}];
                int const offset = ((y.strides()[0] * i) +
                                    (y.strides()[1] * j));
                byte_t* correct_addr = reinterpret_cast<byte_t*>(y.data()) + offset;
                TypeParam const correct_elem = reinterpret_cast<TypeParam*>(correct_addr)[0];
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }
    }

    TYPED_TEST_P(TestNdArrayIndex, TestOperatorParenthesis1) {
        namespace ndu = util;

        ndarray<TypeParam> x({2, 6, 7});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }
        ASSERT_EQ(x.base().use_count(), 1);

        // Slice into a contiguous array. =====================================
        std::vector<ndu::slice> spec_1 {ndu::slice(),
                                        ndu::slice(0, 20, 3)};
        auto y = x(spec_1);

        ASSERT_EQ(y.base(), x.base());
        ASSERT_EQ(y.base().use_count(), 2);
        ASSERT_EQ(y.data(), x.data());
        ASSERT_EQ(y.shape(), shape_t({2, 2, 7}));
        using stride_tt = stride_t::value_type;
        auto expected_strides_y = stride_t({
                                    1 * 6 * 7 * static_cast<stride_tt>(sizeof(TypeParam)),
                                    3 *     7 * static_cast<stride_tt>(sizeof(TypeParam)),
                                    1 *         static_cast<stride_tt>(sizeof(TypeParam))});
        ASSERT_EQ(y.strides(), expected_strides_y);

        std::vector<TypeParam> correct_y { 0,  1,  2,  3,  4,  5,  6,
                                          21, 22, 23, 24, 25, 26, 27,
                                          42, 43, 44, 45, 46, 47, 48,
                                          63, 64, 65, 66, 67, 68, 69};
        size_t offset_1 = 0;
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k) {
                    TypeParam const tested_elem = y[{i, j, k}];
                    TypeParam const correct_elem = correct_y[offset_1];
                    ASSERT_EQ(tested_elem, correct_elem);
                    ++offset_1;
                }
            }
        }

        // Slice into a non-contiguous array ==================================
        std::vector<ndu::slice> spec_2 {ndu::slice(1, 2),
                                        ndu::slice(),
                                        ndu::slice(2, 6, 3)};
        auto z = y(spec_2);

        ASSERT_EQ(z.base(), x.base());
        ASSERT_EQ(z.base().use_count(), 3);
        ASSERT_EQ(z.data(), x.data() + 44);
        ASSERT_EQ(z.shape(), shape_t({1, 2, 2}));
        auto expected_strides_z = stride_t({expected_strides_y[0],
                                            expected_strides_y[1],
                                            expected_strides_y[2] * spec_2[2].step()});
        ASSERT_EQ(z.strides(), expected_strides_z);

        std::vector<TypeParam> correct_z {44, 47,
                                          65, 68};
        size_t offset_2 = 0;
        for(size_t i = 0; i < z.shape()[0]; ++i) {
            for(size_t j = 0; j < z.shape()[1]; ++j) {
                for(size_t k = 0; k < z.shape()[2]; ++k) {
                    TypeParam const tested_elem = z[{i, j, k}];
                    TypeParam const correct_elem = correct_z[offset_2];
                    ASSERT_EQ(tested_elem, correct_elem);
                    ++offset_2;
                }
            }
        }
    }

    TYPED_TEST_P(TestNdArrayIndex, TestOperatorParenthesis2) {
        namespace ndu = util;

        ndarray<TypeParam> x({2, 6, 7});
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }
        ASSERT_EQ(x.base().use_count(), 1);

        // Slice into a contiguous array. =====================================
        std::vector<ndu::slice> spec_1 {ndu::slice(),
                                        ndu::slice(10, 0, -2), // try degenerate(0, -1, -2)
                                        ndu::slice(4, 0, -1)};
        auto y = x(spec_1);

        ASSERT_EQ(y.base(), x.base());
        ASSERT_EQ(y.base().use_count(), 2);
        ASSERT_EQ(y.data(), x.data() + 39);
        ASSERT_EQ(y.shape(), shape_t({2, 3, 4}));
        using stride_tt = stride_t::value_type;
        auto expected_strides_y = stride_t({
                                    1 * 6 * 7 * static_cast<stride_tt>(sizeof(TypeParam)),
                                   -2 *     7 * static_cast<stride_tt>(sizeof(TypeParam)),
                                   -1 *         static_cast<stride_tt>(sizeof(TypeParam))});
        ASSERT_EQ(y.strides(), expected_strides_y);

        std::vector<TypeParam> correct_y {39, 38, 37, 36,
                                          25, 24, 23, 22,
                                          11, 10,  9,  8,
                                          81, 80, 79, 78,
                                          67, 66, 65, 64,
                                          53, 52, 51, 50};
        size_t offset_1 = 0;
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k) {
                    TypeParam const tested_elem = y[{i, j, k}];
                    TypeParam const correct_elem = correct_y[offset_1];
                    ASSERT_EQ(tested_elem, correct_elem);
                    ++offset_1;
                }
            }
        }

        // Slice into a non-contiguous array ==================================
        std::vector<ndu::slice> spec_2 {ndu::slice(2, -1, -1),
                                        ndu::slice(1, 2),
                                        ndu::slice(2, 0, -2)};
        auto z = y(spec_2);

        ASSERT_EQ(z.base(), x.base());
        ASSERT_EQ(z.base().use_count(), 3);
        ASSERT_EQ(z.data(), x.data() + 65);
        ASSERT_EQ(z.shape(), shape_t({2, 1, 1}));
        auto expected_strides_z = stride_t({-1 * expected_strides_y[0],
                                                 expected_strides_y[1],
                                            -2 * expected_strides_y[2]});
        ASSERT_EQ(z.strides(), expected_strides_z);

        std::vector<TypeParam> correct_z {65, 23};
        size_t offset_2 = 0;
        for(size_t i = 0; i < z.shape()[0]; ++i) {
            for(size_t j = 0; j < z.shape()[1]; ++j) {
                for(size_t k = 0; k < z.shape()[2]; ++k) {
                    TypeParam const tested_elem = z[{i, j, k}];
                    TypeParam const correct_elem = correct_z[offset_2];
                    ASSERT_EQ(tested_elem, correct_elem);
                    ++offset_2;
                }
            }
        }
    }

    /*
     * We only test integer types here to keep testing code simple.
     * Index/subsetting should not be affected by types anyway.
     */
    typedef ::testing::Types<int, size_t> MyTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayIndex,
                               TestOperatorSquareBracket,
                               TestOperatorParenthesis1,
                               TestOperatorParenthesis2);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayIndex, MyTypes);
}

#endif // TEST_NDARRAY_INDEX_CPP
