// ############################################################################
// test_ndarray_property.cpp
// =========================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_PROPERTY_CPP
#define TEST_NDARRAY_PROPERTY_CPP

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T>
    class TestNdArrayProperty : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayProperty);

    TYPED_TEST_P(TestNdArrayProperty, TestSize) {
        ndarray<TypeParam> x(shape_t({2, 3, 4}));

        ASSERT_EQ(x.size(), 2 * 3 * 4);
    }

    TYPED_TEST_P(TestNdArrayProperty, TestNDim) {
        ndarray<TypeParam> x(shape_t({5}));
        ndarray<TypeParam> y(shape_t({2, 3, 4}));

        ASSERT_EQ(x.ndim(), 1);
        ASSERT_EQ(y.ndim(), 3);
    }

    TYPED_TEST_P(TestNdArrayProperty, TestNBytes) {
        ndarray<TypeParam> x(shape_t({5}));
        ndarray<TypeParam> y(shape_t({2, 3, 4}));

        ASSERT_EQ(x.nbytes(), 5 * sizeof(TypeParam));
        ASSERT_EQ(y.nbytes(), 2 * 3 * 4 * sizeof(TypeParam));
    }

    TYPED_TEST_P(TestNdArrayProperty, TestEquals) {
        ndarray<TypeParam> w(shape_t({5}));
        ndarray<TypeParam> x(shape_t({2, 3, 4}));
        ndarray<TypeParam> y(shape_t({2, 3, 4}));
        ndarray<TypeParam> z(y);

        ASSERT_FALSE(w.equals(x)); ASSERT_FALSE(x.equals(w));
        ASSERT_FALSE(w.equals(y)); ASSERT_FALSE(y.equals(w));
        ASSERT_FALSE(w.equals(z)); ASSERT_FALSE(z.equals(w));

        ASSERT_FALSE(x.equals(y)); ASSERT_FALSE(y.equals(x));
        ASSERT_FALSE(x.equals(z)); ASSERT_FALSE(z.equals(x));

        ASSERT_TRUE(y.equals(z));  ASSERT_TRUE(z.equals(y));
    }

    typedef ::testing::Types<bool, int, size_t,
                             float, double,
                             std::complex<float>,
                             std::complex<double>> MyTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayProperty,
                               TestSize,
                               TestNDim,
                               TestNBytes,
                               TestEquals);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayProperty, MyTypes);
}

#endif // TEST_NDARRAY_PROPERTY_CPP
