// ############################################################################
// test_ndarray_iterator.cpp
// =========================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_ITERATOR_CPP
#define TEST_NDARRAY_ITERATOR_CPP

#include <vector>

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T>
    class TestNdArrayIterator : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayIterator);

    TYPED_TEST_P(TestNdArrayIterator, TestOperatorEqualEqual) {
        ndarray<TypeParam> x(shape_t({5}));
        ndarray_iterator<TypeParam> x_iter1(&x);
        ndarray_iterator<TypeParam> x_iter2(&x);
        ndarray_iterator<TypeParam> x_iter3(x_iter1);
        ASSERT_TRUE(x_iter1 == x_iter2);
        ASSERT_TRUE(x_iter1 == x_iter3);

        TypeParam* data_y = new TypeParam[5*3*4];
        shape_t const shape_y({5, 3, 4});
        ndarray<TypeParam> y(reinterpret_cast<byte_t*>(data_y), shape_y);
        ndarray_iterator<TypeParam> y_iter(&y);
        ASSERT_FALSE(x_iter1 == y_iter);
    }

    TYPED_TEST_P(TestNdArrayIterator, TestOperatorEqual) {
        ndarray<TypeParam> x(shape_t({5}));
        ndarray_iterator<TypeParam> x_iter(&x);
        ndarray<TypeParam> y(shape_t({5, 3, 4}));
        ndarray_iterator<TypeParam> y_iter(&y);

        ASSERT_FALSE(x_iter == y_iter);
        y_iter = x_iter;
        ASSERT_TRUE(x_iter == y_iter);
    }

    TYPED_TEST_P(TestNdArrayIterator, TestOperatorDereference) {
        ndarray<TypeParam> x(shape_t({5, 3, 4}));
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        for(size_t i = 0; i < x.shape()[0]; ++i) {
            for(size_t j = 0; j < x.shape()[1]; ++j) {
                for(size_t k = 0; k < x.shape()[2]; ++k) {
                    ndarray_iterator<TypeParam> x_iter(&x, {i, j, k});
                    TypeParam const& correct_elem = x[{i,j,k}];
                    ASSERT_EQ(*x_iter, correct_elem);
                }
            }
        }
    }

    TYPED_TEST_P(TestNdArrayIterator, TestOperatorIncrement) {
        ndarray<TypeParam> x(shape_t({50, 25, 30}));
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        ndarray_iterator<TypeParam> x_iter(&x);
        for(size_t i = 0; i < x.size(); ++i, ++x_iter) {
            TypeParam const& correct_elem = x.data()[i];
            TypeParam const& tested_elem = *x_iter;
            ASSERT_EQ(tested_elem, correct_elem);
        }
        // Iterator fully consumed:
        ndarray_iterator<TypeParam> expected_x_iter(&x, x.shape());
        ASSERT_TRUE(x_iter == expected_x_iter);

        namespace ndu = util;
        ndarray<TypeParam> y = x({ndu::slice(),
                                  ndu::slice(0, 30, 2),
                                  ndu::slice(20, 13, -3)});
        ndarray_iterator<TypeParam> y_iter(&y);
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k) {
                    TypeParam const& correct_elem = y[{i,j,k}];
                    TypeParam const& tested_elem = *y_iter;
                    ASSERT_EQ(tested_elem, correct_elem);
                    ++y_iter;
                }
            }
        }
        // Iterator fully consumed:
        ndarray_iterator<TypeParam> expected_y_iter(&y, y.shape());
        ASSERT_TRUE(y_iter == expected_y_iter);
    }

    TYPED_TEST_P(TestNdArrayIterator, TestNdArrayIteration) {
        ndarray<TypeParam> x(shape_t({5, 3, 4}));
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        size_t i = 0;
        for(auto it = x.begin(); it != x.end(); ++it, ++i) {
            TypeParam const& tested_elem = *it;
            TypeParam const& correct_elem = x.data()[i];
            ASSERT_EQ(tested_elem, correct_elem);
        }

        namespace ndu = util;
        ndarray<TypeParam> y = x({ndu::slice(5, -1, -2)});
        auto it = y.begin();
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k, ++it) {
                    TypeParam const& tested_elem = *it;
                    TypeParam const& correct_elem = y[{i,j,k}];
                    ASSERT_EQ(tested_elem, correct_elem);
                }
            }
        }
    }

    TYPED_TEST_P(TestNdArrayIterator, TestAssignment) {
        ndarray<TypeParam> x(shape_t({5, 3, 4}));
        for(size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<TypeParam>(i);
        }

        for(auto it = x.begin(); it != x.end(); ++it) {
            *it += TypeParam(2.0);
        }

        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& tested_elem = x.data()[i];
            TypeParam const& correct_elem = TypeParam(i + 2.0);
            ASSERT_EQ(tested_elem, correct_elem);
        }
    }

    typedef ::testing::Types<bool, int, size_t,
                             float, double,
                             std::complex<float>,
                             std::complex<double>> MyTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayIterator,
                               TestOperatorEqualEqual,
                               TestOperatorEqual,
                               TestOperatorDereference,
                               TestOperatorIncrement,
                               TestNdArrayIteration,
                               TestAssignment);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayIterator, MyTypes);
}

#endif // TEST_NDARRAY_ITERATOR_CPP
