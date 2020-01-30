// ############################################################################
// test_ndarray.cpp
// ================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_CPP
#define TEST_NDARRAY_CPP

#include <complex>
#include <numeric>
#include <stdexcept>

#include <gtest/gtest.h>
#include "test_type.hpp"

#include "ndarray/ndarray.hpp"

namespace nd {
    /*
     * Note
     * ----
     * We mostly test ndarray<T> when T=int for test readability.
     * When we have doubts on whether a method will work for another type, then
     * we test those explicitly.
     */

    template <typename T>
    class TestNdArrayArithmetic : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayArithmetic);



    TYPED_TEST_P(TestNdArrayArithmetic, TestConstructorShape) {
        shape_t const shape({1, 2, 3});
        ndarray<TypeParam> x(shape);

        ASSERT_NE(x.data(), nullptr);
        ASSERT_EQ(x.shape(), shape);
        using stride_tt = stride_t::value_type;
        auto expected_strides = stride_t({
                                  static_cast<stride_tt>(6 * sizeof(TypeParam)),
                                  static_cast<stride_tt>(3 * sizeof(TypeParam)),
                                  static_cast<stride_tt>(sizeof(TypeParam))});
        ASSERT_EQ(x.strides(), expected_strides);
        ASSERT_TRUE(x.is_contiguous());
    }

    TYPED_TEST_P(TestNdArrayArithmetic, TestConstructorCopyExplicit) {
        shape_t const shape({1, 2, 3});
        ndarray<TypeParam> x(shape);
        ASSERT_EQ(x.base().use_count(), 1);

        ndarray<TypeParam> y(x.base(),
                             reinterpret_cast<byte_t*>(x.data()),
                             x.shape(),
                             x.strides());

        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(x.data(), y.data());
        ASSERT_EQ(x.shape(), y.shape());
        ASSERT_EQ(x.strides(), y.strides());
        ASSERT_EQ(x.base().use_count(), 2);
        ASSERT_TRUE(y.is_contiguous());
    }

    TYPED_TEST_P(TestNdArrayArithmetic, TestConstructorCopy) {
        size_t const nelem = 50;
        auto x = new ndarray<TypeParam>({nelem});
        ASSERT_EQ((x->base()).use_count(), 1);
        for(size_t i = 0; i < nelem; ++i) {
            x->data()[i] = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y(*x);
        ASSERT_EQ((x->base()).use_count(), 2);
        ASSERT_EQ(y.base().use_count(), 2);

        delete x;
        ASSERT_EQ(y.base().use_count(), 1);
        for(size_t i = 0; i < y.size(); ++i) {
            ASSERT_EQ(y.data()[i], TypeParam(i));
        }
    }

    TYPED_TEST_P(TestNdArrayArithmetic, TestConstructorMove) {
        ndarray<TypeParam> x({5, 3, 4});
        ASSERT_FALSE(x.shape().empty());
        ASSERT_FALSE(x.strides().empty());

        shape_t shape = x.shape();
        stride_t strides = x.strides();

        ndarray<TypeParam> z(std::move(x));
        ASSERT_EQ(x.base().use_count(), 2);
        ASSERT_TRUE(x.shape().empty());
        ASSERT_TRUE(x.strides().empty());
        ASSERT_EQ(shape, z.shape());
        ASSERT_EQ(strides, z.strides());
    }

    TYPED_TEST_P(TestNdArrayArithmetic, TestConstructorPointerAndShape) {
        shape_t const shape({3, 4, 5});
        size_t const nelem = 3 * 4 * 5;
        TypeParam* data = new TypeParam[nelem];
        for(size_t i = 0; i < nelem; ++i) {
            data[i] = static_cast<TypeParam>(i);
        }

        {
            ndarray<TypeParam> x(reinterpret_cast<byte_t*>(data), shape);

            ASSERT_EQ(x.base(), nullptr);
            ASSERT_EQ(x.data(), data);
            ASSERT_EQ(x.shape(), shape);
            using stride_tt = stride_t::value_type;
            auto expected_strides = stride_t({
                                      static_cast<stride_tt>(20 * sizeof(TypeParam)),
                                      static_cast<stride_tt>(5  * sizeof(TypeParam)),
                                      static_cast<stride_tt>(sizeof(TypeParam))});
            ASSERT_EQ(x.strides(), expected_strides);
        }

        for(size_t i = 0; i < nelem; ++i) {
            ASSERT_EQ(data[i], static_cast<TypeParam>(i));
        }
    }

    TYPED_TEST_P(TestNdArrayArithmetic, TestConstructorPointerAndShapeAndStride) {
        size_t const nelem = 6 * 8 * 10;
        TypeParam* data = new TypeParam[nelem];
        for(size_t i = 0; i < nelem; ++i) {
            data[i] = static_cast<TypeParam>(i);
        }

        // if A = np.arange(6*8*10).reshape(6,8,10), then we are constructing
        // the shape/strides for A[::2, ::2, ::2].
        shape_t const shape({3, 4, 5});
        using stride_tt = stride_t::value_type;
        auto strides = stride_t({
                         static_cast<stride_tt>(2 * 80 * sizeof(TypeParam)),
                         static_cast<stride_tt>(2 * 10  * sizeof(TypeParam)),
                         static_cast<stride_tt>(2 * sizeof(TypeParam))});
        ndarray<TypeParam> x(reinterpret_cast<byte_t*>(data), shape, strides);

        ASSERT_EQ(x.base(), nullptr);
        ASSERT_EQ(x.data(), data);
        ASSERT_EQ(x.shape(), shape);
        ASSERT_EQ(x.strides(), strides);
        ASSERT_FALSE(x.is_contiguous());

        // check that a given cell holds correct element.
        size_t const offset = ((strides[0] * 1) +
                               (strides[1] * 2) +
                               (strides[2] * 3));
        ASSERT_EQ(reinterpret_cast<TypeParam*>(
                  reinterpret_cast<byte_t*>(x.data()) + offset)[0],
                  static_cast<TypeParam>(206));
    }

    TEST(TestNdArray, TestSize) {
        ndarray<int> x({2, 3, 4});
        ASSERT_EQ(x.size(), 2u * 3u * 4u);
    }

    TEST(TestNdArray, TestNDim) {
       {ndarray<int> x({5});
        ASSERT_EQ(x.ndim(), 1u);}

       {ndarray<int> x({2, 3, 4});
        ASSERT_EQ(x.ndim(), 3u);}
    }

    TEST(TestNdArray, TestNBytes) {
       {ndarray<int> x({5});
        ASSERT_EQ(x.nbytes(), 5 * sizeof(int));}

       {ndarray<int> x({2, 3, 4});
        ASSERT_EQ(x.nbytes(), 2 * 3 * 4 * sizeof(int));}
    }

    TEST(TestNdArray, TestEquals) {
        ndarray<int> w({5});
        ndarray<int> x({2, 3, 4});
        ndarray<int> y({2, 3, 4});
        ndarray<int> z(y);
        ndarray<int> t = y;

        ASSERT_FALSE(w.equals(x)); ASSERT_FALSE(x.equals(w));
        ASSERT_FALSE(w.equals(y)); ASSERT_FALSE(y.equals(w));
        ASSERT_FALSE(w.equals(z)); ASSERT_FALSE(z.equals(w));

        ASSERT_FALSE(x.equals(y)); ASSERT_FALSE(y.equals(x));
        ASSERT_FALSE(x.equals(z)); ASSERT_FALSE(z.equals(x));

        ASSERT_TRUE(y.equals(z));  ASSERT_TRUE(z.equals(y));

        ASSERT_TRUE(t.equals(y)); ASSERT_TRUE(y.equals(t));
    }

    TEST(TestNdArray, TestOperatorSquareBracket) {
        // Positive strides
        ndarray<int> x({3, 1, 4, 5});
        std::iota(x.begin(), x.end(), 0);

        for(size_t i = 0; i < x.shape()[0]; ++i) {
            for(size_t j = 0; j < x.shape()[1]; ++j) {
                for(size_t k = 0; k < x.shape()[2]; ++k) {
                    for(size_t l = 0; l < x.shape()[3]; ++l) {
                        int const tested_elem = x[{i, j, k, l}];
                        int const offset = ((x.strides()[0] * i) +
                                            (x.strides()[1] * j) +
                                            (x.strides()[2] * k) +
                                            (x.strides()[3] * l));
                        byte_t* correct_addr = reinterpret_cast<byte_t*>(x.data()) + offset;
                        int const correct_elem = reinterpret_cast<int*>(correct_addr)[0];
                        ASSERT_EQ(tested_elem, correct_elem);
                    }
                }
            }
        }

        /* Can update entries. */
        int& tested_elem = x[{2, 0, 3, 4}];
        int const correct_elem = 500;
        tested_elem = correct_elem;
        ASSERT_EQ(tested_elem, correct_elem);

        // Negative strides
        shape_t shape({5, 3});
        stride_t strides({-3 * static_cast<int>(sizeof(int)),
                               static_cast<int>(sizeof(int))});
        int* data = new int[5 * 3];
        int* data_end = data + 15;
        std::iota(data, data_end, 0);
        ndarray<int> y(reinterpret_cast<byte_t*>(data_end - 1), shape, strides);
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                int const tested_elem = y[{i, j}];
                int const offset = ((y.strides()[0] * i) +
                                    (y.strides()[1] * j));
                byte_t* correct_addr = reinterpret_cast<byte_t*>(y.data()) + offset;
                int const correct_elem = reinterpret_cast<int*>(correct_addr)[0];
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }
    }

    TEST(TestNdArray, TestAt) {
        // Positive strides
        ndarray<int> x({3, 1, 4, 5});
        std::iota(x.begin(), x.end(), 0);

        for(size_t i = 0; i < x.shape()[0]; ++i) {
            for(size_t j = 0; j < x.shape()[1]; ++j) {
                for(size_t k = 0; k < x.shape()[2]; ++k) {
                    for(size_t l = 0; l < x.shape()[3]; ++l) {
                        int const tested_elem = x.at({i, j, k, l});
                        int const offset = ((x.strides()[0] * i) +
                                            (x.strides()[1] * j) +
                                            (x.strides()[2] * k) +
                                            (x.strides()[3] * l));
                        byte_t* correct_addr = reinterpret_cast<byte_t*>(x.data()) + offset;
                        int const correct_elem = reinterpret_cast<int*>(correct_addr)[0];
                        ASSERT_EQ(tested_elem, correct_elem);
                    }
                }
            }
        }

        // Out of bounds access
        ASSERT_NO_THROW((x.at({0, 0, 0, 0})));
        ASSERT_NO_THROW((x.at({2, 0, 3, 4})));
        ASSERT_THROW((x.at({3, 0, 3, 4})), std::runtime_error);
        ASSERT_THROW((x.at({2, 1, 3, 4})), std::runtime_error);
        ASSERT_THROW((x.at({2, 0, 4, 4})), std::runtime_error);
        ASSERT_THROW((x.at({2, 0, 3, 5})), std::runtime_error);

        /* Can update entries. */
        int& tested_elem = x.at({2, 0, 3, 4});
        int const correct_elem = 500;
        tested_elem = correct_elem;
        ASSERT_EQ(tested_elem, correct_elem);

        // Negative strides
        shape_t shape({5, 3});
        stride_t strides({-3 * static_cast<int>(sizeof(int)),
                               static_cast<int>(sizeof(int))});
        int* data = new int[5 * 3];
        int* data_end = data + 15;
        std::iota(data, data_end, 0);
        ndarray<int> y(reinterpret_cast<byte_t*>(data_end - 1), shape, strides);
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                int const tested_elem = y.at({i, j});
                int const offset = ((y.strides()[0] * i) +
                                    (y.strides()[1] * j));
                byte_t* correct_addr = reinterpret_cast<byte_t*>(y.data()) + offset;
                int const correct_elem = reinterpret_cast<int*>(correct_addr)[0];
                ASSERT_EQ(tested_elem, correct_elem);
            }
        }
    }

    TEST(TestNdArray, TestOperatorParenthesis1) {
        ndarray<int> x({2, 6, 7});
        std::iota(x.begin(), x.end(), 0);
        ASSERT_EQ(x.base().use_count(), 1);

        // Slice into a contiguous array. =====================================
        std::vector<util::slice> s1({{}, {0, 20, 3}});
        auto y = x(s1);

        ASSERT_EQ(y.base(), x.base());
        ASSERT_EQ(y.base().use_count(), 2);
        ASSERT_EQ(y.data(), x.data());
        ASSERT_EQ(y.shape(), shape_t({2, 2, 7}));
        auto expected_strides_y = stride_t({ 1 * 6 * 7 * sizeof(int),
                                             3 *     7 * sizeof(int),
                                             1 *         sizeof(int)});
        ASSERT_EQ(y.strides(), expected_strides_y);

        auto gt_y = (r_<int>({ 0,  1,  2,  3,  4,  5,  6,
                              21, 22, 23, 24, 25, 26, 27,
                              42, 43, 44, 45, 46, 47, 48,
                              63, 64, 65, 66, 67, 68, 69})
                     .reshape({2, 2, 7}));
        ASSERT_TRUE(allclose(y, gt_y));

        // Slice into a non-contiguous array ==================================
        std::vector<util::slice> s2({{1, 2}, {}, {2, 6, 3}});
        auto z = y(s2);

        ASSERT_EQ(z.base(), x.base());
        ASSERT_EQ(z.base().use_count(), 3);
        ASSERT_EQ(z.data(), x.data() + 44);
        ASSERT_EQ(z.shape(), shape_t({1, 2, 2}));
        auto expected_strides_z = stride_t({expected_strides_y[0],
                                            expected_strides_y[1],
                                            expected_strides_y[2] * s2[2].step()});
        ASSERT_EQ(z.strides(), expected_strides_z);

        auto gt_z = (r_<int>({44, 47,
                              65, 68})
                     .reshape({1, 2, 2}));
        ASSERT_TRUE(allclose(z, gt_z));
    }

    TEST(TestNdArray, TestOperatorParenthesis2) {
        ndarray<int> x({2, 6, 7});
        std::iota(x.begin(), x.end(), 0);
        ASSERT_EQ(x.base().use_count(), 1);

        // Slice into a contiguous array. =====================================
        std::vector<util::slice> s1({{}, {10, 0, -2}, {4, 0, -1}});
        auto y = x(s1);

        ASSERT_EQ(y.base(), x.base());
        ASSERT_EQ(y.base().use_count(), 2);
        ASSERT_EQ(y.data(), x.data() + 39);
        ASSERT_EQ(y.shape(), shape_t({2, 3, 4}));
        auto expected_strides_y = stride_t({ 1 * 6 * 7 * sizeof(int),
                                            -2 *     7 * static_cast<int>(sizeof(int)),
                                            -1 *         static_cast<int>(sizeof(int))});
        ASSERT_EQ(y.strides(), expected_strides_y);

        auto gt_y = (r_<int>({39, 38, 37, 36,
                              25, 24, 23, 22,
                              11, 10,  9,  8,
                              81, 80, 79, 78,
                              67, 66, 65, 64,
                              53, 52, 51, 50})
                     .reshape({2, 3, 4}));
        ASSERT_TRUE(allclose(y, gt_y));

        // Slice into a non-contiguous array ==================================
        std::vector<util::slice> s2({{2, -1, -1}, {1, 2}, {2, 0, -2}});
        auto z = y(s2);

        ASSERT_EQ(z.base(), x.base());
        ASSERT_EQ(z.base().use_count(), 3);
        ASSERT_EQ(z.data(), x.data() + 65);
        ASSERT_EQ(z.shape(), shape_t({2, 1, 1}));
        auto expected_strides_z = stride_t({-1 * expected_strides_y[0],
                                                 expected_strides_y[1],
                                            -2 * expected_strides_y[2]});
        ASSERT_EQ(z.strides(), expected_strides_z);

        auto gt_z = (r_<int>({65, 23})
                     .reshape({2, 1, 1}));
        ASSERT_TRUE(allclose(z, gt_z));
    }

    TEST(TestNdArray, TestWhere) {
        // shape(mask) == shape(array)
       {ndarray<bool> mask = (r_<bool>({true, false, true, false})
                              .reshape({2, 2}));
        auto x = (arange<int>(0, mask.size(), 1)
                  .reshape({2, 2}));
        auto y = x.where(mask);
        auto gt = r_<int>({0, 2});
        ASSERT_TRUE(allclose(y, gt));}

        // shape(mask) broadcasts
       {ndarray<bool> mask = nd::r_<bool>({true, false});
        auto x = (arange<int>(0, 4, 1)
                  .reshape({2, 2}));
        auto y = x.where(mask);
        auto gt = r_<int>({0, 2});
        ASSERT_TRUE(allclose(y, gt));}
    }

    TEST(TestNdArray, TestFilter) {
        // shape(mask) == shape(array), scalar x
       {ndarray<bool> mask = (r_<bool>({true, false, true, false})
                              .reshape({2, 2}));
        auto ar = (arange<int>(0, mask.size(), 1)
                   .reshape({2, 2}));
        int const x = 1;
        auto gt = (r_<int>({1, 1, 1, 3})
                   .reshape({2, 2}));

        auto const& ar2 = ar.filter(mask, x);
        ASSERT_TRUE(ar2.equals(ar));
        ASSERT_TRUE(allclose(ar, gt));}

        // shape(mask) == shape(array), vector x
       {ndarray<bool> mask = (r_<bool>({true, false, true, false})
                              .reshape({2, 2}));
        auto ar = (arange<int>(0, mask.size(), 1)
                   .reshape({2, 2}));
        auto x = arange<int>(0, 2, 1) * 2 + 1;
        auto gt = (r_<int>({1, 1, 3, 3})
                   .reshape({2, 2}));

        auto const& ar2 = ar.filter(mask, x);
        ASSERT_TRUE(ar2.equals(ar));
        ASSERT_TRUE(allclose(ar, gt));}

        // shape(mask) broadcasts, scalar x
       {ndarray<bool> mask = r_<bool>({true, false});
        auto ar = (arange<int>(0, 4, 1)
                   .reshape({2, 2}));
        int const x = 1;
        auto gt = (r_<int>({1, 1, 1, 3})
                   .reshape({2, 2}));

        auto const& ar2 = ar.filter(mask, x);
        ASSERT_TRUE(ar2.equals(ar));
        ASSERT_TRUE(allclose(ar, gt));}

        // shape(mask) broadcasts, vector x
       {ndarray<bool> mask = r_<bool>({true, false});
        auto ar = (arange<int>(0, 4, 1)
                   .reshape({2, 2}));
        auto x = arange<int>(0, 2, 1) * 2 + 1;
        auto gt = (r_<int>({1, 1, 3, 3})
                   .reshape({2, 2}));

        auto const& ar2 = ar.filter(mask, x);
        ASSERT_TRUE(ar2.equals(ar));
        ASSERT_TRUE(allclose(ar, gt));}
    }

    TEST(TestNdArray, TestBeginEnd) {
        auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4}));

        size_t i = 0;
        for(auto it = x.begin(); it != x.end(); ++it, ++i) {
            int const& tested_elem = *it;
            int const& correct_elem = x.data()[i];
            ASSERT_EQ(tested_elem, correct_elem);
        }

        auto y = x({{5, -1, -2}});
        auto it = y.begin();
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k, ++it) {
                    int const& tested_elem = *it;
                    int const& correct_elem = y[{i,j,k}];
                    ASSERT_EQ(tested_elem, correct_elem);
                }
            }
        }
    }

    TEST(TestNdArray, TestCopy) {
        // Contiguous array
        auto x = (arange<int>(0, 20 * 30 * 40, 1)
                  .reshape({20, 30, 40}));

       {auto x_cpy = x.copy();
        ASSERT_EQ(x.base().use_count(), 1);
        ASSERT_EQ(x_cpy.base().use_count(), 1);
        ASSERT_FALSE(x_cpy.equals(x));
        ASSERT_TRUE(x_cpy.is_contiguous());
        ASSERT_EQ(x_cpy.shape(), shape_t({20, 30, 40}));
        ASSERT_TRUE(allclose(x, x_cpy));}


        // Strided array
       {auto y = x({{20, 0, -5}, {0, 5, 2}, {10, 2, -3}});
        auto y_cpy = y.copy();
        auto gt = (r_<int>({22810, 22807, 22804,
                            22890, 22887, 22884,
                            22970, 22967, 22964,

                            16810, 16807, 16804,
                            16890, 16887, 16884,
                            16970, 16967, 16964,

                            10810, 10807, 10804,
                            10890, 10887, 10884,
                            10970, 10967, 10964,

                             4810,  4807,  4804,
                             4890,  4887,  4884,
                             4970,  4967,  4964})
                   .reshape({4, 3, 3}));

        ASSERT_EQ(y_cpy.base().use_count(), 1);
        ASSERT_FALSE(y_cpy.equals(y));
        ASSERT_TRUE(y_cpy.is_contiguous());
        ASSERT_EQ(y_cpy.shape(), shape_t({4, 3, 3}));
        ASSERT_TRUE(allclose(gt, y_cpy));}
    }

    TEST(TestNdArray, TestAsContiguousArray) {
        // Contiguous array
        ndarray<int> x({20, 30, 40});
        auto x_cont = ascontiguousarray(x);
        ASSERT_TRUE(x.equals(x_cont));

        // Strided array
        auto y = x({{50, -1, -2}});
        auto y_cont = ascontiguousarray(y);
        ASSERT_FALSE(y.is_contiguous());
        ASSERT_TRUE(y_cont.is_contiguous());
        ASSERT_TRUE(allclose(y, y_cont));
    }

    TEST(TestNdArray, TestSqueeze) {
       {auto x = r_<int>({1});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {auto x = r_<int>({5});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({5, 3});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5, 3}));
        ASSERT_EQ(y.strides(), stride_t({ 3 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 3});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 3});
        auto y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({5, 1});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({5, 1});
        auto y = x.squeeze({1});
        ASSERT_EQ(y.shape(), shape_t({5}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({5, 3, 4});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5, 3, 4}));
        ASSERT_EQ(y.strides(), stride_t({12 * sizeof(int),
                                          4 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 3, 4});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({3, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 3, 4});
        auto y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({3, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({5, 1, 4});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({5, 1, 4});
        auto y = x.squeeze({1});
        ASSERT_EQ(y.shape(), shape_t({5, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({5, 3, 1});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5, 3}));
        ASSERT_EQ(y.strides(), stride_t({ 3 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({5, 3, 1});
        auto y = x.squeeze({2});
        ASSERT_EQ(y.shape(), shape_t({5, 3}));
        ASSERT_EQ(y.strides(), stride_t({ 3 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 1, 4});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({4}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 1, 4});
        auto y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({1, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 1, 4});
        auto y = x.squeeze({1});
        ASSERT_EQ(y.shape(), shape_t({1, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 1, 4});
        auto y = x.squeeze({0, 1});
        ASSERT_EQ(y.shape(), shape_t({4}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 3, 1});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 3, 1});
        auto y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({3, 1}));
        ASSERT_EQ(y.strides(), stride_t({ 1 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 3, 1});
        auto y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({3, 1}));
        ASSERT_EQ(y.strides(), stride_t({ 1 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 3, 1});
        auto y = x.squeeze({2});
        ASSERT_EQ(y.shape(), shape_t({1, 3}));
        ASSERT_EQ(y.strides(), stride_t({ 3 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 3, 1});
        auto y = x.squeeze({0, 2});
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 1, 1});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 1, 1});
        auto y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({1, 1}));
        ASSERT_EQ(y.strides(), stride_t({ 1 * sizeof(int),
                                              sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 1, 1});
        auto y = x.squeeze({0, 1});
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<int> x({1, 1, 1});
        auto y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        ASSERT_EQ(x.base().use_count(), 2);}
    }

    TEST(TestNdArray, TestReshape) {
        // Contiguous array
       {auto x = arange<int>(0, 10 * 10 * 5, 1);
        auto y = x.reshape({10, 10, 5});
        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(y.shape(), shape_t({10, 10, 5}));
        ASSERT_EQ(y.strides(), stride_t({10 * 5 * sizeof(int),
                                              5 * sizeof(int),
                                                  sizeof(int)}));
        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, *it_y);
        }}

        // Strided Array
       {auto x = zeros<int>({10, 20, 30})({{10, -1, -2}});
        std::iota(x.begin(), x.end(), 0);
        auto y = x.reshape({5, 10, 60});
        ASSERT_NE(x.base(), y.base());
        ASSERT_EQ(y.shape(), shape_t({5, 10, 60}));
        ASSERT_EQ(y.strides(), stride_t({10 * 60 * sizeof(int),
                                              60 * sizeof(int),
                                                   sizeof(int)}));
        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, *it_y);
        }}
    }

    TEST(TestNdArray, TestRavel) {
        // Contiguous array
       {auto x = (arange<int>(0, 10 * 20 * 30, 1)
                  .reshape({10, 20, 30}));

        auto y = x.ravel();
        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(y.shape(), shape_t({10 * 20 * 30}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, *it_y);
        }}

        // Strided array
       {auto x = zeros<int>({10, 20, 30})({{10, -1, -2}});
        std::iota(x.begin(), x.end(), 0);

        auto y = x.ravel();
        ASSERT_NE(x.base(), y.base());
        ASSERT_EQ(y.shape(), shape_t({5 * 20 * 30}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(int)}));
        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, *it_y);
        }}
    }

    TEST(TestNdArray, TestBroadcastTo) {
        // Contiguous Array
       {auto x = (arange<int>(0, 2 * 3 * 4, 1)
                  .reshape({2, 3, 4}));
        auto y = x.broadcast_to({5, 6, 2, 3, 4});
        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(x.data(), y.data());
        ASSERT_EQ(y.shape(), shape_t({5, 6, 2, 3, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 0 * sizeof(int),
                                          0 * sizeof(int),
                                         12 * sizeof(int),
                                          4 * sizeof(int),
                                              sizeof(int)}));
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k) {
                    for(size_t l = 0; l < y.shape()[3]; ++l) {
                        for(size_t m = 0; m < y.shape()[4]; ++m) {
                            auto const& tested_elem = y[{i, j, k, l, m}];
                            auto const& correct_elem = x[{k, l, m}];

                            ASSERT_EQ(tested_elem, correct_elem);
                        }
                    }
                }
            }
        }}

       {auto x = (arange<int>(0, 2 * 1 * 4, 1)
                  .reshape({2, 1, 4}));
        auto y = x.broadcast_to({5, 6, 2, 3, 4});
        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(x.data(), y.data());
        ASSERT_EQ(y.shape(), shape_t({5, 6, 2, 3, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 0 * sizeof(int),
                                          0 * sizeof(int),
                                          4 * sizeof(int),
                                          0 * sizeof(int),
                                              sizeof(int)}));
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k) {
                    for(size_t l = 0; l < y.shape()[3]; ++l) {
                        for(size_t m = 0; m < y.shape()[4]; ++m) {
                            int const& tested_elem = y[{i, j, k, l, m}];
                            int const& correct_elem = x[{k, 0, m}];

                            ASSERT_EQ(tested_elem, correct_elem);
                        }
                    }
                }
            }
        }}

       {auto x = (arange<int>(0, 2 * 1 * 4, 1)
                  .reshape({2, 1, 4}));
        auto y = x.broadcast_to({2, 1, 4});
        ASSERT_TRUE(x.equals(y));}
    }

    TEST(TestNdArray, TestTranspose) {
       { // Initially contiguous arrays
        auto x = (arange<int>(0, 2 * 3 * 4, 1)
                  .reshape({2, 3, 4}));
        auto gt012 = x.copy();
        auto gt120 = (r_<int>({ 0, 12,
                                1, 13,
                                2, 14,
                                3, 15,

                                4, 16,
                                5, 17,
                                6, 18,
                                7, 19,

                                8, 20,
                                9, 21,
                               10, 22,
                               11, 23})
                      .reshape({3, 4, 2}));
        auto gt201 = (r_<int>({ 0,  4,  8,
                               12, 16, 20,

                                1,  5,  9,
                               13, 17, 21,

                                2,  6, 10,
                               14, 18, 22,

                                3,  7, 11,
                               15, 19, 23})
                      .reshape({4, 2, 3}));
        auto gt102 = (r_<int>({ 0,  1,  2,  3,
                               12, 13, 14, 15,

                                4,  5,  6,  7,
                               16, 17, 18, 19,

                                8,  9, 10, 11,
                               20, 21, 22, 23})
                      .reshape({3, 2, 4}));
        auto gt021 = (r_<int>({ 0,  4,  8,
                                1,  5,  9,
                                2,  6, 10,
                                3,  7, 11,

                               12, 16, 20,
                               13, 17, 21,
                               14, 18, 22,
                               15, 19, 23})
                      .reshape({2, 4, 3}));
        auto gt210 = (r_<int>({ 0, 12,
                                4, 16,
                                8, 20,

                                1, 13,
                                5, 17,
                                9, 21,

                                2, 14,
                                6, 18,
                               10, 22,

                                3, 15,
                                7, 19,
                               11, 23})
                      .reshape({4, 3, 2}));

        ASSERT_NO_THROW(x.transpose());
        ASSERT_NO_THROW(x.transpose({0, 1, 2}));
        ASSERT_NO_THROW(x.transpose({1, 2, 0}));
        ASSERT_NO_THROW(x.transpose({2, 0, 1}));
        ASSERT_NO_THROW(x.transpose({1, 0, 2}));
        ASSERT_NO_THROW(x.transpose({0, 2, 1}));
        ASSERT_NO_THROW(x.transpose({2, 1, 0}));
        ASSERT_THROW(x.transpose({0, 1, 3}), std::runtime_error);
        ASSERT_THROW(x.transpose({1, 0, 0}), std::runtime_error);

        ASSERT_TRUE(allclose(x.transpose({0, 1, 2}), gt012));
        ASSERT_TRUE(allclose(x.transpose({1, 2, 0}), gt120));
        ASSERT_TRUE(allclose(x.transpose({2, 0, 1}), gt201));
        ASSERT_TRUE(allclose(x.transpose({1, 0, 2}), gt102));
        ASSERT_TRUE(allclose(x.transpose({0, 2, 1}), gt021));
        ASSERT_TRUE(allclose(x.transpose({2, 1, 0}), gt210));
        ASSERT_TRUE(allclose(x.transpose()         , gt210));}

       { // Initially non-contiguous arrays
        auto x = (arange<int>(0, 2 * 3 * 4, 1)
                  .reshape({2, 3, 4})
                  .operator()({{}, {}, {3, -1, -2}}));
        ASSERT_EQ(x.shape(), shape_t({2, 3, 2}));
        ASSERT_FALSE(x.is_contiguous());
        auto gt012 = x.copy();
        auto gt120 = (r_<int>({ 3, 15,
                                1, 13,

                                7, 19,
                                5, 17,

                               11, 23,
                                9, 21})
                      .reshape({3, 2, 2}));
        auto gt201 = (r_<int>({ 3,  7, 11,
                               15, 19, 23,

                                1,  5,  9,
                               13, 17, 21})
                      .reshape({2, 2, 3}));
        auto gt102 = (r_<int>({ 3,  1,
                               15, 13,

                                7,  5,
                               19, 17,

                               11,  9,
                               23, 21})
                      .reshape({3, 2, 2}));
        auto gt021 = (r_<int>({ 3,  7, 11,
                                1,  5,  9,

                               15, 19, 23,
                               13, 17, 21})
                      .reshape({2, 2, 3}));
        auto gt210 = (r_<int>({ 3, 15,
                                7, 19,
                               11, 23,

                                1, 13,
                                5, 17,
                                9, 21})
                      .reshape({2, 3, 2}));

        ASSERT_NO_THROW(x.transpose());
        ASSERT_NO_THROW(x.transpose({0, 1, 2}));
        ASSERT_NO_THROW(x.transpose({1, 2, 0}));
        ASSERT_NO_THROW(x.transpose({2, 0, 1}));
        ASSERT_NO_THROW(x.transpose({1, 0, 2}));
        ASSERT_NO_THROW(x.transpose({0, 2, 1}));
        ASSERT_NO_THROW(x.transpose({2, 1, 0}));
        ASSERT_THROW(x.transpose({0, 1, 3}), std::runtime_error);
        ASSERT_THROW(x.transpose({1, 0, 0}), std::runtime_error);

        ASSERT_TRUE(allclose(x.transpose({0, 1, 2}), gt012));
        ASSERT_TRUE(allclose(x.transpose({1, 2, 0}), gt120));
        ASSERT_TRUE(allclose(x.transpose({2, 0, 1}), gt201));
        ASSERT_TRUE(allclose(x.transpose({1, 0, 2}), gt102));
        ASSERT_TRUE(allclose(x.transpose({0, 2, 1}), gt021));
        ASSERT_TRUE(allclose(x.transpose({2, 1, 0}), gt210));
        ASSERT_TRUE(allclose(x.transpose()         , gt210));}
    }

    TEST(TestNdArray, TestCast) {
        auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4}));
        auto y = x.template cast<float>();
        auto gt = (arange<float>(0, 5 * 3 * 4, 1)
                   .reshape({5, 3, 4}));
        ASSERT_TRUE(allclose(y, gt));
    }



    REGISTER_TYPED_TEST_CASE_P(TestNdArrayArithmetic,
                               TestConstructorShape,
                               TestConstructorCopyExplicit,
                               TestConstructorCopy,
                               TestConstructorMove,
                               TestConstructorPointerAndShape,
                               TestConstructorPointerAndShapeAndStride);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayArithmetic, MyArithmeticTypes);
}

#endif // TEST_NDARRAY_CPP
