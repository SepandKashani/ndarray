// ############################################################################
// test_ndarray_manipulation.cpp
// =============================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_MANIPULATION_CPP
#define TEST_NDARRAY_MANIPULATION_CPP

#include <vector>

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T>
    class TestNdArrayManipulation : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayManipulation);

    TYPED_TEST_P(TestNdArrayManipulation, TestCopy) {
        // Contiguous array
        ndarray<TypeParam> x(shape_t({20, 30, 40}));
        auto x_iter = x.begin();
        for(size_t i = 0; i < x.size(); ++i, ++x_iter) {
            *x_iter = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> x_cpy = x.copy();
        ASSERT_EQ(x.base().use_count(), 1);
        ASSERT_EQ(x_cpy.base().use_count(), 1);

        ASSERT_FALSE(x_cpy.equals(x));
        ASSERT_TRUE(x_cpy.is_contiguous());
        ASSERT_EQ(x_cpy.shape(), shape_t({20, 30, 40}));
        auto x_cpy_iter = x_cpy.begin();
        for(size_t i = 0; i < x_cpy.size(); ++i, ++x_cpy_iter) {
            TypeParam const tested_elem = *x_cpy_iter;
            TypeParam const correct_elem = static_cast<TypeParam>(i);
            ASSERT_EQ(tested_elem, correct_elem);
        }

        // Strided array
        ndarray<TypeParam> y = x({util::slice(20, 0, -5),
                                  util::slice(0, 5, 2),
                                  util::slice(10, 2, -3)});
        ndarray<TypeParam> y_cpy = y.copy();
        ASSERT_EQ(y_cpy.base().use_count(), 1);

        ASSERT_FALSE(y_cpy.equals(y));
        ASSERT_TRUE(y_cpy.is_contiguous());
        ASSERT_EQ(y_cpy.shape(), shape_t({4, 3, 3}));
        auto y_cpy_iter = y_cpy.begin();
        std::vector<TypeParam> correct_y {static_cast<TypeParam>(22810),
                                          static_cast<TypeParam>(22807),
                                          static_cast<TypeParam>(22804),
                                          static_cast<TypeParam>(22890),
                                          static_cast<TypeParam>(22887),
                                          static_cast<TypeParam>(22884),
                                          static_cast<TypeParam>(22970),
                                          static_cast<TypeParam>(22967),
                                          static_cast<TypeParam>(22964),
                                          static_cast<TypeParam>(16810),
                                          static_cast<TypeParam>(16807),
                                          static_cast<TypeParam>(16804),
                                          static_cast<TypeParam>(16890),
                                          static_cast<TypeParam>(16887),
                                          static_cast<TypeParam>(16884),
                                          static_cast<TypeParam>(16970),
                                          static_cast<TypeParam>(16967),
                                          static_cast<TypeParam>(16964),
                                          static_cast<TypeParam>(10810),
                                          static_cast<TypeParam>(10807),
                                          static_cast<TypeParam>(10804),
                                          static_cast<TypeParam>(10890),
                                          static_cast<TypeParam>(10887),
                                          static_cast<TypeParam>(10884),
                                          static_cast<TypeParam>(10970),
                                          static_cast<TypeParam>(10967),
                                          static_cast<TypeParam>(10964),
                                          static_cast<TypeParam>( 4810),
                                          static_cast<TypeParam>( 4807),
                                          static_cast<TypeParam>( 4804),
                                          static_cast<TypeParam>( 4890),
                                          static_cast<TypeParam>( 4887),
                                          static_cast<TypeParam>( 4884),
                                          static_cast<TypeParam>( 4970),
                                          static_cast<TypeParam>( 4967),
                                          static_cast<TypeParam>( 4964)};
        for(size_t i = 0; i < correct_y.size(); ++i, ++y_cpy_iter) {
            ASSERT_EQ(*y_cpy_iter, correct_y[i]);
        }
    }

    TYPED_TEST_P(TestNdArrayManipulation, TestAsContiguousArray) {
        // Contiguous array
        ndarray<TypeParam> x(shape_t({20, 30, 40}));
        ndarray<TypeParam> x_cont = ascontiguousarray(x);
        ASSERT_TRUE(x.equals(x_cont));

        ndarray<TypeParam> y = x({util::slice(50, -1, -2)});
        ndarray<TypeParam> y_cont = ascontiguousarray(y);
        ASSERT_FALSE(y.is_contiguous());
        ASSERT_TRUE(y_cont.is_contiguous());
        for(auto it_y = y.begin(), it_y_cont = y_cont.begin();
            it_y != y.end(); ++it_y, ++it_y_cont) {
            ASSERT_EQ(*it_y, *it_y_cont);
        }
    }

    TYPED_TEST_P(TestNdArrayManipulation, TestSqueeze) {
       {ndarray<TypeParam> x(1);
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(5);
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({5, 3}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5, 3}));
        ASSERT_EQ(y.strides(), stride_t({ 3 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 3}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 3}));
        ndarray<TypeParam> y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({5, 1}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({5, 1}));
        ndarray<TypeParam> y = x.squeeze({1});
        ASSERT_EQ(y.shape(), shape_t({5}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({5, 3, 4}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5, 3, 4}));
        ASSERT_EQ(y.strides(), stride_t({12 * sizeof(TypeParam),
                                          4 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 3, 4}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({3, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 3, 4}));
        ndarray<TypeParam> y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({3, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({5, 1, 4}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({5, 1, 4}));
        ndarray<TypeParam> y = x.squeeze({1});
        ASSERT_EQ(y.shape(), shape_t({5, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({5, 3, 1}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({5, 3}));
        ASSERT_EQ(y.strides(), stride_t({ 3 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({5, 3, 1}));
        ndarray<TypeParam> y = x.squeeze({2});
        ASSERT_EQ(y.shape(), shape_t({5, 3}));
        ASSERT_EQ(y.strides(), stride_t({ 3 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 1, 4}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({4}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 1, 4}));
        ndarray<TypeParam> y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({1, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 1, 4}));
        ndarray<TypeParam> y = x.squeeze({1});
        ASSERT_EQ(y.shape(), shape_t({1, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 4 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 1, 4}));
        ndarray<TypeParam> y = x.squeeze({0, 1});
        ASSERT_EQ(y.shape(), shape_t({4}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 3, 1}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 3, 1}));
        ndarray<TypeParam> y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({3, 1}));
        ASSERT_EQ(y.strides(), stride_t({ 1 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 3, 1}));
        ndarray<TypeParam> y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({3, 1}));
        ASSERT_EQ(y.strides(), stride_t({ 1 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 3, 1}));
        ndarray<TypeParam> y = x.squeeze({2});
        ASSERT_EQ(y.shape(), shape_t({1, 3}));
        ASSERT_EQ(y.strides(), stride_t({ 3 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 3, 1}));
        ndarray<TypeParam> y = x.squeeze({0, 2});
        ASSERT_EQ(y.shape(), shape_t({3}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 1, 1}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 1, 1}));
        ndarray<TypeParam> y = x.squeeze({0});
        ASSERT_EQ(y.shape(), shape_t({1, 1}));
        ASSERT_EQ(y.strides(), stride_t({ 1 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 1, 1}));
        ndarray<TypeParam> y = x.squeeze({0, 1});
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}

       {ndarray<TypeParam> x(shape_t({1, 1, 1}));
        ndarray<TypeParam> y = x.squeeze();
        ASSERT_EQ(y.shape(), shape_t({1}));
        ASSERT_EQ(y.strides(), stride_t({sizeof(TypeParam)}));
        ASSERT_EQ(x.base().use_count(), 2);}
    }

    TYPED_TEST_P(TestNdArrayManipulation, TestReshape) {
        // Contiguous array
       {ndarray<TypeParam> x(shape_t({500}));
        auto it = x.begin();
        for(size_t i = 0; it != x.end(); ++i, ++it) {
            *it = static_cast<TypeParam>(i);
        }
        ndarray<TypeParam> y = x.reshape({10, 10, 5});
        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(y.shape(), shape_t({10, 10, 5}));
        ASSERT_EQ(y.strides(), stride_t({10 * 5 * static_cast<int>(sizeof(TypeParam)),
                                              5 * static_cast<int>(sizeof(TypeParam)),
                                                  static_cast<int>(sizeof(TypeParam))}));
        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, *it_y);
        }}

        // Strided Array
       {ndarray<TypeParam> _x(shape_t({10, 20, 30}));
        ndarray<TypeParam> x = _x({util::slice(10, -1, -2)});
        auto it = x.begin();
        for(size_t i = 0; it != x.end(); ++i, ++it) {
            *it = static_cast<TypeParam>(i);
        }
        ndarray<TypeParam> y = x.reshape({5, 10, 60});
        ASSERT_NE(x.base(), y.base());
        ASSERT_EQ(y.shape(), shape_t({5, 10, 60}));
        ASSERT_EQ(y.strides(), stride_t({10 * 60 * static_cast<int>(sizeof(TypeParam)),
                                              60 * static_cast<int>(sizeof(TypeParam)),
                                                   static_cast<int>(sizeof(TypeParam))}));
        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, *it_y);
        }}
    }

    TYPED_TEST_P(TestNdArrayManipulation, TestRavel) {
        // Contiguous array
       {ndarray<TypeParam> x(shape_t({10, 20, 30}));
        auto it = x.begin();
        for(size_t i = 0; it != x.end(); ++i, ++it) {
            *it = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y = x.ravel();
        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(y.shape(), shape_t({10 * 20 * 30}));
        ASSERT_EQ(y.strides(), stride_t({static_cast<int>(sizeof(TypeParam))}));
        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, *it_y);
        }}

        // Strided array
       {ndarray<TypeParam> _x(shape_t({10, 20, 30}));
        ndarray<TypeParam> x = _x({util::slice(10, -1, -2)});
        auto it = x.begin();
        for(size_t i = 0; it != x.end(); ++i, ++it) {
            *it = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y = x.ravel();
        ASSERT_NE(x.base(), y.base());
        ASSERT_EQ(y.shape(), shape_t({5 * 20 * 30}));
        ASSERT_EQ(y.strides(), stride_t({static_cast<int>(sizeof(TypeParam))}));
        for(auto it_x = x.begin(), it_y = y.begin();
            it_x != x.end();
            ++it_x, ++it_y) {
            ASSERT_EQ(*it_x, *it_y);
        }}
    }

    TYPED_TEST_P(TestNdArrayManipulation, TestBroadcastTo) {
        // Contiguous Array
       {ndarray<TypeParam> x(shape_t({2, 3, 4}));
        auto it = x.begin();
        for(size_t i = 0; it != x.end(); ++it, ++i) {
            *it = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y = x.broadcast_to(shape_t({5, 6, 2, 3, 4}));
        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(x.data(), y.data());
        ASSERT_EQ(y.shape(), shape_t({5, 6, 2, 3, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 0 * sizeof(TypeParam),
                                          0 * sizeof(TypeParam),
                                         12 * sizeof(TypeParam),
                                          4 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k) {
                    for(size_t l = 0; l < y.shape()[3]; ++l) {
                        for(size_t m = 0; m < y.shape()[4]; ++m) {
                            TypeParam const& tested_elem = y[{i,j,k,l,m}];
                            TypeParam const& correct_elem = x[{k, l, m}];

                            ASSERT_EQ(tested_elem, correct_elem);
                        }
                    }
                }
            }
        }}

       {ndarray<TypeParam> x(shape_t({2, 1, 4}));
        auto it = x.begin();
        for(size_t i = 0; it != x.end(); ++it, ++i) {
            *it = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y = x.broadcast_to(shape_t({5, 6, 2, 3, 4}));
        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(x.data(), y.data());
        ASSERT_EQ(y.shape(), shape_t({5, 6, 2, 3, 4}));
        ASSERT_EQ(y.strides(), stride_t({ 0 * sizeof(TypeParam),
                                          0 * sizeof(TypeParam),
                                          4 * sizeof(TypeParam),
                                          0 * sizeof(TypeParam),
                                              sizeof(TypeParam)}));
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k) {
                    for(size_t l = 0; l < y.shape()[3]; ++l) {
                        for(size_t m = 0; m < y.shape()[4]; ++m) {
                            TypeParam const& tested_elem = y[{i,j,k,l,m}];
                            TypeParam const& correct_elem = x[{k, 0, m}];

                            ASSERT_EQ(tested_elem, correct_elem);
                        }
                    }
                }
            }
        }}

       {ndarray<TypeParam> x(shape_t({2, 1, 4}));
        auto it = x.begin();
        for(size_t i = 0; it != x.end(); ++it, ++i) {
            *it = static_cast<TypeParam>(i);
        }

        ndarray<TypeParam> y = x.broadcast_to(shape_t({2, 1, 4}));
        ASSERT_TRUE(x.equals(y));}
    }

    typedef ::testing::Types<bool, int, size_t,
                             float, double,
                             std::complex<float>,
                             std::complex<double>> MyTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayManipulation,
                               TestCopy,
                               TestAsContiguousArray,
                               TestSqueeze,
                               TestReshape,
                               TestRavel,
                               TestBroadcastTo);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayManipulation, MyTypes);
}

#endif // TEST_NDARRAY_MANIPULATION_CPP
