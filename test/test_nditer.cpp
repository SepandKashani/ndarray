// ############################################################################
// test_nditer.cpp
// ===============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDITER_CPP
#define TEST_NDITER_CPP

#include <vector>

#include <gtest/gtest.h>
#include "test_type.hpp"

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T>
    class TestNdArrayIter : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayIter);

    TYPED_TEST_P(TestNdArrayIter, TestOperatorEqualEqual) {
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

    TYPED_TEST_P(TestNdArrayIter, TestOperatorEqual) {
        ndarray<TypeParam> x(shape_t({5}));
        ndarray_iterator<TypeParam> x_iter(&x);
        ndarray<TypeParam> y(shape_t({5, 3, 4}));
        ndarray_iterator<TypeParam> y_iter(&y);

        ASSERT_FALSE(x_iter == y_iter);
        y_iter = x_iter;
        ASSERT_TRUE(x_iter == y_iter);
    }

    TYPED_TEST_P(TestNdArrayIter, TestOperatorDereference) {
        ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

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

    TYPED_TEST_P(TestNdArrayIter, TestOperatorIncrement) {
        ndarray<TypeParam> x = (arange<int>(0, 50 * 25 * 30, 1)
                                .reshape(shape_t({50, 25, 30}))
                                .template cast<TypeParam>());

        ndarray_iterator<TypeParam> x_iter(&x);
        for(size_t i = 0; i < x.size(); ++i, ++x_iter) {
            TypeParam const& correct_elem = x.data()[i];
            TypeParam const& tested_elem = *x_iter;
            ASSERT_EQ(tested_elem, correct_elem);
        }
        // Iterator fully consumed:
        ndarray_iterator<TypeParam> expected_x_iter(&x, x.shape());
        ASSERT_TRUE(x_iter == expected_x_iter);

        ndarray<TypeParam> y = x({util::slice(),
                                  util::slice(0, 30, 2),
                                  util::slice(20, 13, -3)});
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

    TYPED_TEST_P(TestNdArrayIter, TestAssignment) {
        ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        for(auto it = x.begin(); it != x.end(); ++it) {
            *it += TypeParam(2.0);
        }

        for(size_t i = 0; i < x.size(); ++i) {
            TypeParam const& tested_elem = x.data()[i];
            TypeParam const& correct_elem = TypeParam(i + 2.0);
            ASSERT_EQ(tested_elem, correct_elem);
        }
    }

    TYPED_TEST_P(TestNdArrayIter, TestSTL) {
        // check if ndarray_iterator<T> is compatible with STL libraries.
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape(shape_t({5, 3, 4}))
                  .template cast<TypeParam>());
        auto y = ndarray<TypeParam>(shape_t{x.size()});
        std::copy(x.begin(), x.end(), y.begin());

        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            TypeParam const& gt = *it_x;
            TypeParam const& res = *it_y;
            ASSERT_EQ(gt, res);
        }}
    }

    REGISTER_TYPED_TEST_CASE_P(TestNdArrayIter,
                               TestOperatorEqualEqual,
                               TestOperatorEqual,
                               TestOperatorDereference,
                               TestOperatorIncrement,
                               TestAssignment,
                               TestSTL);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayIter, MyArithmeticTypes);
}

#endif // TEST_NDITER_CPP
