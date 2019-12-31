// ############################################################################
// test_ndvector.cpp
// =================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDVECTOR_CPP
#define TEST_NDVECTOR_CPP

#include <numeric>
#include <stdexcept>
#include <vector>

#include <gtest/gtest.h>
#include "test_type.hpp"

#include "ndarray/ndarray.hpp"

namespace nd {
    TEST(TestNdVector, TestDefaultConstructor) {
        vector<int> x;
        ASSERT_EQ(x.size(), 0u);
    }

    TEST(TestNdVector, TestConstructorCount) {
       {vector<int> x(0);
        ASSERT_EQ(x.size(), 0u);}

       {vector<int> x(1);
        ASSERT_EQ(x.size(), 1u);}

        ASSERT_THROW(vector<int>(9), std::runtime_error);
    }

    TEST(TestNdVector, TestConstructorCountValue) {
       {vector<int> x(3u, 5);
        ASSERT_EQ(x.size(), 3u);
        ASSERT_EQ(x[0], 5);
        ASSERT_EQ(x[1], 5);
        ASSERT_EQ(x[2], 5);}

        ASSERT_NO_THROW(vector<int>(8));
        ASSERT_THROW(vector<int>(9), std::runtime_error);
    }

    TEST(TestNdVector, TestConstructorIter) {
       {std::vector<int> x(5);
        std::iota(x.begin(), x.end(), 0);
        vector<int> y(x.begin(), x.end());
        ASSERT_EQ(y.size(), 5u);
        ASSERT_EQ(y[0], 0);
        ASSERT_EQ(y[1], 1);
        ASSERT_EQ(y[2], 2);
        ASSERT_EQ(y[3], 3);
        ASSERT_EQ(y[4], 4);}

       {std::vector<int> x(5);
        std::iota(x.begin(), x.end(), 0);
        vector<int> y(x.rbegin(), x.rend());
        ASSERT_EQ(y.size(), 5u);
        ASSERT_EQ(y[0], 4);
        ASSERT_EQ(y[1], 3);
        ASSERT_EQ(y[2], 2);
        ASSERT_EQ(y[3], 1);
        ASSERT_EQ(y[4], 0);}

       {std::vector<int> x(9);
        std::iota(x.begin(), x.end(), 0);
        ASSERT_THROW(vector<int>(x.rbegin(), x.rend()), std::runtime_error);}
    }

    TEST(TestNdVector, TestConstructorCopy) {
       {vector<int> x(3u, 1);
        vector<int> y(x);

        ASSERT_EQ(x.size(), y.size());
        ASSERT_EQ(x[0], y[0]);
        ASSERT_EQ(x[1], y[1]);
        ASSERT_EQ(x[2], y[2]);
        ASSERT_NE(&x[0], &y[0]);}
    }

    TEST(TestNdVector, TestConstructorIList) {
       {vector<int> x = {1, 2, 3, 4};
        ASSERT_EQ(x.size(), 4u);
        ASSERT_EQ(x[0], 1);
        ASSERT_EQ(x[1], 2);
        ASSERT_EQ(x[2], 3);
        ASSERT_EQ(x[3], 4);}

       {vector<int> x({1, 2, 3, 4});
        ASSERT_EQ(x.size(), 4u);
        ASSERT_EQ(x[0], 1);
        ASSERT_EQ(x[1], 2);
        ASSERT_EQ(x[2], 3);
        ASSERT_EQ(x[3], 4);}

        ASSERT_THROW((vector<int, 3>({0, 1, 2, 3})), std::runtime_error);
    }

    TEST(TestNdVector, TestOperatorEqualCopy) {
       {vector<int> x = {1, 2, 3};
        vector<int> y = {4, 5};
        auto& z = (x = y);
        ASSERT_EQ(x.size(), 2u);
        ASSERT_EQ(x[0], 4);
        ASSERT_EQ(x[1], 5);
        ASSERT_EQ(&x[0], &z[0]);}
    }

    TEST(TestNdVector, TestOperatorEqualIList) {
       {vector<int> x = {1, 2, 3};
        auto& z = (x = {4, 5});
        ASSERT_EQ(x.size(), 2u);
        ASSERT_EQ(x[0], 4);
        ASSERT_EQ(x[1], 5);
        ASSERT_EQ(&x[0], &z[0]);}

       {vector<int, 2> x = {1, 2};
        ASSERT_THROW((x = {1, 2, 3}), std::runtime_error);}
    }

    TEST(TestNdVector, TestOperatorSquareBracket) {
       {vector<int> x = {1, 2, 3};
        ASSERT_EQ(x[0], 1);
        (x[0] = 5) = 7;
        ASSERT_EQ(x.size(), 3u);
        ASSERT_EQ(x[0], 7);
        ASSERT_NO_THROW(x[5]);} // Works, but no idea what is at that location.
    }

    TEST(TestNdVector, TestData) {
       {vector<int> x = {1, 2, 3};
        ASSERT_EQ(x.data()[0], 1);
        ASSERT_EQ(*(x.data() + 1), 2);}
    }

    TEST(TestNdVector, TestBegin) {
       {vector<int> x = {1, 2, 3};
        ASSERT_EQ(&(*x.begin()), x.data());
        ASSERT_EQ(*(x.begin() + 1), 2);}
    }

    TEST(TestNdVector, TestEnd) {
       {vector<int> x = {1, 2, 3};
        ASSERT_EQ(*(x.end() - 1), 3);
        ASSERT_EQ(*(x.end() - 2), 2);}
    }

    TEST(TestNdVector, TestRBegin) {
       {vector<int> x = {1, 2, 3};
        ASSERT_EQ(*x.rbegin(), 3);
        ASSERT_EQ(*(x.rbegin() + 1), 2);
        ASSERT_EQ(*(x.rbegin() + 2), 1);}
    }

    TEST(TestNdVector, TestREnd) {
       {vector<int> x = {1, 2, 3};
        ASSERT_EQ(*(x.rend() - 1), 1);
        ASSERT_EQ(*(x.rend() - 2), 2);
        ASSERT_EQ(*(x.rend() - 3), 3);
        ASSERT_EQ(x.rend() - 3, x.rbegin());}
    }

    TEST(TestNdVector, TestEmpty) {
       {vector<int> x(3u);
        ASSERT_FALSE(x.empty());}

       {vector<int> x;
        ASSERT_TRUE(x.empty());}
    }

    TEST(TestNdVector, TestSize) {
       {vector<int> x(3u);
        ASSERT_EQ(x.size(), 3u);}

       {vector<int> x;
        ASSERT_EQ(x.size(), 0u);}
    }

    TEST(TestNdVector, TestMaxSize) {
       {size_t constexpr N_MAX = 8;
        vector<int, N_MAX> x;
        ASSERT_EQ(x.max_size(), N_MAX);}

       {size_t constexpr N_MAX = 3;
        vector<int, N_MAX> x;
        ASSERT_EQ(x.max_size(), N_MAX);}
    }

    TEST(TestNdVector, TestClear) {
       {vector<int> x;
        ASSERT_EQ(x.size(), 0u);
        x.clear();
        ASSERT_EQ(x.size(), 0u);}

       {vector<int> x = {1, 2, 3};
        ASSERT_EQ(x.size(), 3u);
        x.clear();
        ASSERT_EQ(x.size(), 0u);}
    }

    TEST(TestNdVector, TestInsert) {
        // Full arrays
       {vector<int, 3u> x(3u);
        for(auto it = x.begin() - 1; it != x.end() + 1; ++it) {
            ASSERT_THROW(x.insert(it, -1), std::runtime_error);
        }}

        // Insertion at the beginning
       {vector<int> x;
        x.insert(x.begin(), -1);
        ASSERT_TRUE(x == vector<int>({-1}));
        x.insert(x.begin(), -2);
        ASSERT_TRUE(x == vector<int>({-2, -1}));
        x.insert(x.begin(), -3);
        ASSERT_TRUE(x == vector<int>({-3, -2, -1}));}

        // Insertion at the tail
       {vector<int> x;
        x.insert(x.end(), -1);
        ASSERT_TRUE(x == vector<int>({-1}));
        x.insert(x.end(), -2);
        ASSERT_TRUE(x == vector<int>({-1, -2}));
        x.insert(x.end(), -3);
        ASSERT_TRUE(x == vector<int>({-1, -2, -3}));}

        // Insertion in the middle
       {vector<int> x = {1, 2};
        x.insert(x.begin() + 1, -1);
        ASSERT_TRUE(x == vector<int>({1, -1, 2}));
        x.insert(x.end() - 1, -2);
        ASSERT_TRUE(x == vector<int>({1, -1, -2, 2}));}

        // Correct return value
       {vector<int> x = {1, 2};
        auto const y = x.insert(x.begin(), -1);
        ASSERT_EQ(*y, -1);
        auto const z = x.insert(x.end() - 1, -2);
        ASSERT_EQ(*z, -2);
        auto const t = x.insert(x.end(), -3);
        ASSERT_EQ(*t, -3);}
    }

    TEST(TestNdVector, TestPushBack) {
       {vector<int> x;
        for(size_t i = 0; i < x.max_size(); ++i) {
            ASSERT_EQ(x.size(), i);
            x.push_back(static_cast<int>(i));
            ASSERT_EQ(x[i], static_cast<int>(i));
        }
        ASSERT_EQ(x.size(), x.max_size());
        ASSERT_THROW(x.push_back(-1), std::runtime_error);}

       {vector<int> x = {1, 2, 3};
        ASSERT_EQ(x.size(), 3u);
        x.push_back(-1);
        ASSERT_EQ(x[3], -1);
        ASSERT_EQ(x.size(), 4u);}
    }

    TEST(TestNdVector, TestPopBack) {
       {vector<int> x(3u);
        std::iota(x.begin(), x.end(), 1);
        ASSERT_EQ(x.size(), 3u);
        x.push_back(-3);
        ASSERT_EQ(x.size(), 4u);
        x.pop_back();
        ASSERT_EQ(x.size(), 3u);
        x.pop_back();
        x.pop_back();
        ASSERT_EQ(x.size(), 1u);
        x.pop_back();
        ASSERT_EQ(x.size(), 0u);
        x.pop_back();
        ASSERT_EQ(x.size(), 0u);}
    }

    TEST(TestNdVector, TestOperatorEqualEqual) {
       ASSERT_TRUE(vector<int>() == vector<int>());
       ASSERT_TRUE(vector<int>({1,}) == vector<int>({1,}));
       ASSERT_FALSE(vector<int>({2,}) == vector<int>({1,}));
       ASSERT_FALSE(vector<int>({1, 2}) == vector<int>({1,}));
    }

    TEST(TestNdVector, TestOperatorNotEqual) {
       ASSERT_FALSE(vector<int>() != vector<int>());
       ASSERT_FALSE(vector<int>({1,}) != vector<int>({1,}));
       ASSERT_TRUE(vector<int>({2,}) != vector<int>({1,}));
       ASSERT_TRUE(vector<int>({1, 2}) != vector<int>({1,}));
    }
}

#endif // TEST_NDVECTOR_CPP
