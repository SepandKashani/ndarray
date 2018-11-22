// ############################################################################
// test_ndarray_constructor.cpp
// ============================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_CONSTRUCTOR_CPP
#define TEST_NDARRAY_CONSTRUCTOR_CPP

#include <complex>
#include <vector>

#include <gtest/gtest.h>

#include "ndarray/ndarray.hpp"

namespace nd {
    template <typename T>
    class TestNdArrayConstructor : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayConstructor);

    TYPED_TEST_P(TestNdArrayConstructor, TestScalar) {
        TypeParam scalar = TypeParam(1.0);

        ndarray<TypeParam> x(scalar);

        ASSERT_EQ(x.data()[0], scalar);
        ASSERT_EQ(x.shape(), shape_t({1}));

        using stride_tt = stride_t::value_type;
        auto expected_strides = stride_t({static_cast<stride_tt>(sizeof(TypeParam))});
        ASSERT_EQ(x.strides(), expected_strides);
    }

    TYPED_TEST_P(TestNdArrayConstructor, TestShape) {
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

    TYPED_TEST_P(TestNdArrayConstructor, TestCopyExplicit) {
        shape_t const shape({1, 2, 3});
        ndarray<TypeParam> x(shape);
        ASSERT_EQ(x.base().use_count(), 1);

        ndarray<TypeParam> y(x.base(), reinterpret_cast<byte_t*>(x.data()), x.shape(), x.strides());

        ASSERT_EQ(x.base(), y.base());
        ASSERT_EQ(x.data(), y.data());
        ASSERT_EQ(x.shape(), y.shape());
        ASSERT_EQ(x.strides(), y.strides());
        ASSERT_EQ(x.base().use_count(), 2);
        ASSERT_TRUE(y.is_contiguous());
    }

    TYPED_TEST_P(TestNdArrayConstructor, TestCopy) {
        size_t const nelem = 50;
        auto x = new ndarray<TypeParam>(shape_t({nelem}));
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

    TYPED_TEST_P(TestNdArrayConstructor, TestPointerAndShape) {
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

    TYPED_TEST_P(TestNdArrayConstructor, TestPointerAndShapeAndStride) {
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

    typedef ::testing::Types<bool, int, size_t,
                             float, double,
                             std::complex<float>,
                             std::complex<double>> MyTypes;
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayConstructor,
                               TestScalar,
                               TestShape,
                               TestCopyExplicit,
                               TestCopy,
                               TestPointerAndShape,
                               TestPointerAndShapeAndStride);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayConstructor, MyTypes);
}

#endif // TEST_NDARRAY_CONSTRUCTOR_CPP
