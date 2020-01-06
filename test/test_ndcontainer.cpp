// ############################################################################
// test_ndcontainer.cpp
// ====================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDCONTAINER_CPP
#define TEST_NDCONTAINER_CPP

#include <numeric>

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    TEST(TestNdContainer, TestConstructorSized) {
        size_t const nbytes = 3;
        ndcontainer x(nbytes);

        ASSERT_EQ(x.nbytes(), nbytes);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_TRUE(x.own_memory());
    }

    TEST(TestNdContainer, TestConstructorPreExisting) {
        size_t const nbytes = 3;
        byte_t* data = new byte_t[nbytes];
        std::iota(data, data + nbytes, byte_t(0));

        {
            ndcontainer x(data, nbytes);

            ASSERT_EQ(x.nbytes(), nbytes);
            ASSERT_EQ(x.data(), data);
            ASSERT_FALSE(x.own_memory());
        }

        // `data` still valid after `x` destroyed.
        for(size_t i = 0; i < nbytes; ++i) {
            ASSERT_EQ(data[i], static_cast<byte_t>(i));
        }
    }

    TEST(TestNdContainer, TestDataAlignment) {
        size_t const nbytes = 201;
        ndcontainer x(nbytes);

        uintptr_t mask = reinterpret_cast<uintptr_t>(byte_alignment - 1);
        uintptr_t data = reinterpret_cast<uintptr_t>(x.data());
        ASSERT_EQ(uintptr_t(0), mask & data);
    }
}

#endif // TEST_NDCONTAINER_CPP
