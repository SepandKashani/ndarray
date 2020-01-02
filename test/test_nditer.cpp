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
    TEST(TestNdArrayIter, TestOperatorEqualEqual) {
        ndarray<int> x({5});
        ndarray_iterator<int> x_iter1(&x);
        ndarray_iterator<int> x_iter2(&x);
        ndarray_iterator<int> x_iter3(x_iter1);
        ASSERT_TRUE(x_iter1 == x_iter2);
        ASSERT_TRUE(x_iter1 == x_iter3);

        int* data_y = new int[5*3*4];
        shape_t const shape_y({5, 3, 4});
        ndarray<int> y(reinterpret_cast<byte_t*>(data_y), shape_y);
        ndarray_iterator<int> y_iter(&y);
        ASSERT_FALSE(x_iter1 == y_iter);
    }

    TEST(TestNdArrayIter, TestOperatorEqual) {
        ndarray<int> x({5});
        ndarray_iterator<int> x_iter(&x);
        ndarray<int> y({5, 3, 4});
        ndarray_iterator<int> y_iter(&y);

        ASSERT_FALSE(x_iter == y_iter);
        y_iter = x_iter;
        ASSERT_TRUE(x_iter == y_iter);
    }

    TEST(TestNdArrayIter, TestOperatorDereference) {
        auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4}));

        for(size_t i = 0; i < x.shape()[0]; ++i) {
            for(size_t j = 0; j < x.shape()[1]; ++j) {
                for(size_t k = 0; k < x.shape()[2]; ++k) {
                    ndarray_iterator<int> x_iter(&x, {i, j, k});
                    int const& correct_elem = x[{i,j,k}];
                    ASSERT_EQ(*x_iter, correct_elem);
                }
            }
        }
    }

    TEST(TestNdArrayIter, TestOperatorIncrement) {
        auto x = (arange<int>(0, 50 * 25 * 30, 1)
                  .reshape({50, 25, 30}));

        ndarray_iterator<int> x_iter(&x);
        for(size_t i = 0; i < x.size(); ++i, ++x_iter) {
            int const& correct_elem = x.data()[i];
            int const& tested_elem = *x_iter;
            ASSERT_EQ(tested_elem, correct_elem);
        }
        // Iterator fully consumed:
        ndarray_iterator<int> expected_x_iter(&x, x.shape());
        ASSERT_TRUE(x_iter == expected_x_iter);

        ndarray<int> y = x({util::slice(),
                            util::slice(0, 30, 2),
                            util::slice(20, 13, -3)});
        ndarray_iterator<int> y_iter(&y);
        for(size_t i = 0; i < y.shape()[0]; ++i) {
            for(size_t j = 0; j < y.shape()[1]; ++j) {
                for(size_t k = 0; k < y.shape()[2]; ++k) {
                    int const& correct_elem = y[{i,j,k}];
                    int const& tested_elem = *y_iter;
                    ASSERT_EQ(tested_elem, correct_elem);
                    ++y_iter;
                }
            }
        }
        // Iterator fully consumed:
        ndarray_iterator<int> expected_y_iter(&y, y.shape());
        ASSERT_TRUE(y_iter == expected_y_iter);
    }

    TEST(TestNdArrayIter, TestAssignment) {
        auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4}));

        for(auto it = x.begin(); it != x.end(); ++it) {
            *it += int(2.0);
        }

        for(size_t i = 0; i < x.size(); ++i) {
            int const& tested_elem = x.data()[i];
            int const& correct_elem = int(i + 2.0);
            ASSERT_EQ(tested_elem, correct_elem);
        }
    }

    TEST(TestNdArrayIter, TestSTL) {
        // check if ndarray_iterator<T> is compatible with STL libraries.
       {auto x = (arange<int>(0, 5 * 3 * 4, 1)
                  .reshape({5, 3, 4}));
        auto y = ndarray<int>({x.size()});
        std::copy(x.begin(), x.end(), y.begin());

        for(auto it_x = x.begin(), it_y = y.begin(); it_x != x.end(); ++it_x, ++it_y) {
            int const& gt = *it_x;
            int const& res = *it_y;
            ASSERT_EQ(gt, res);
        }}
    }
}

#endif // TEST_NDITER_CPP
