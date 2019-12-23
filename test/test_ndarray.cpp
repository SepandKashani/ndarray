// ############################################################################
// test_ndarray.cpp
// ================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef TEST_NDARRAY_CPP
#define TEST_NDARRAY_CPP

#include <complex>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "test_type.hpp"

#include "ndarray/ndarray.hpp"

namespace nd {
    /* Constructors ======================================================== */
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

    REGISTER_TYPED_TEST_CASE_P(TestNdArrayConstructor,
                               TestScalar,
                               TestShape,
                               TestCopyExplicit,
                               TestCopy,
                               TestPointerAndShape,
                               TestPointerAndShapeAndStride);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayConstructor, MyArithmeticTypes);



    /* Property ============================================================ */
    template <typename T>
    class TestNdArrayProperty : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayProperty);

    TYPED_TEST_P(TestNdArrayProperty, TestSize) {
        ndarray<TypeParam> x(shape_t({2, 3, 4}));

        ASSERT_EQ(x.size(), size_t(2 * 3 * 4));
    }

    TYPED_TEST_P(TestNdArrayProperty, TestNDim) {
        ndarray<TypeParam> x(shape_t({5}));
        ndarray<TypeParam> y(shape_t({2, 3, 4}));

        ASSERT_EQ(x.ndim(), size_t(1));
        ASSERT_EQ(y.ndim(), size_t(3));
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
        ndarray<TypeParam> t = y;

        ASSERT_FALSE(w.equals(x)); ASSERT_FALSE(x.equals(w));
        ASSERT_FALSE(w.equals(y)); ASSERT_FALSE(y.equals(w));
        ASSERT_FALSE(w.equals(z)); ASSERT_FALSE(z.equals(w));

        ASSERT_FALSE(x.equals(y)); ASSERT_FALSE(y.equals(x));
        ASSERT_FALSE(x.equals(z)); ASSERT_FALSE(z.equals(x));

        ASSERT_TRUE(y.equals(z));  ASSERT_TRUE(z.equals(y));

        ASSERT_TRUE(t.equals(y)); ASSERT_TRUE(y.equals(t));
    }

    REGISTER_TYPED_TEST_CASE_P(TestNdArrayProperty,
                               TestSize,
                               TestNDim,
                               TestNBytes,
                               TestEquals);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayProperty, MyArithmeticTypes);



    /* Index / Filter / Iterate ============================================ */
    template <typename T>
    class TestNdArrayIndex : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayIndex);

    TYPED_TEST_P(TestNdArrayIndex, TestOperatorSquareBracket) {
        // Positive strides
        ndarray<TypeParam> x(shape_t({3, 1, 4, 5}));
        std::iota(x.begin(), x.end(), TypeParam(0));

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
        TypeParam* data_end = data + 15;
        std::iota(data, data_end, TypeParam(0));
        ndarray<TypeParam> y(reinterpret_cast<byte_t*>(data_end - 1), shape, strides);
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
        ndarray<TypeParam> x(shape_t({2, 6, 7}));
        std::iota(x.begin(), x.end(), TypeParam(0));
        ASSERT_EQ(x.base().use_count(), 1);

        // Slice into a contiguous array. =====================================
        std::vector<util::slice> spec_1({util::slice(),
                                         util::slice(0, 20, 3)});
        auto y = x(spec_1);

        ASSERT_EQ(y.base(), x.base());
        ASSERT_EQ(y.base().use_count(), 2);
        ASSERT_EQ(y.data(), x.data());
        ASSERT_EQ(y.shape(), shape_t({2, 2, 7}));
        using stride_tt = stride_t::value_type;
        auto expected_strides_y = stride_t({ 1 * 6 * 7 * static_cast<stride_tt>(sizeof(TypeParam)),
                                             3 *     7 * static_cast<stride_tt>(sizeof(TypeParam)),
                                             1 *         static_cast<stride_tt>(sizeof(TypeParam))});
        ASSERT_EQ(y.strides(), expected_strides_y);

        std::vector<TypeParam> correct_y({ 0,  1,  2,  3,  4,  5,  6,
                                          21, 22, 23, 24, 25, 26, 27,
                                          42, 43, 44, 45, 46, 47, 48,
                                          63, 64, 65, 66, 67, 68, 69});
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
        std::vector<util::slice> spec_2({util::slice(1, 2),
                                         util::slice(),
                                         util::slice(2, 6, 3)});
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
        ndarray<TypeParam> x(shape_t({2, 6, 7}));
        std::iota(x.begin(), x.end(), TypeParam(0));
        ASSERT_EQ(x.base().use_count(), 1);

        // Slice into a contiguous array. =====================================
        std::vector<util::slice> spec_1({util::slice(),
                                         util::slice(10, 0, -2), // try degenerate(0, -1, -2)
                                         util::slice(4, 0, -1)});
        auto y = x(spec_1);

        ASSERT_EQ(y.base(), x.base());
        ASSERT_EQ(y.base().use_count(), 2);
        ASSERT_EQ(y.data(), x.data() + 39);
        ASSERT_EQ(y.shape(), shape_t({2, 3, 4}));
        using stride_tt = stride_t::value_type;
        auto expected_strides_y = stride_t({ 1 * 6 * 7 * static_cast<stride_tt>(sizeof(TypeParam)),
                                            -2 *     7 * static_cast<stride_tt>(sizeof(TypeParam)),
                                            -1 *         static_cast<stride_tt>(sizeof(TypeParam))});
        ASSERT_EQ(y.strides(), expected_strides_y);

        std::vector<TypeParam> correct_y({39, 38, 37, 36,
                                          25, 24, 23, 22,
                                          11, 10,  9,  8,
                                          81, 80, 79, 78,
                                          67, 66, 65, 64,
                                          53, 52, 51, 50});
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
        std::vector<util::slice> spec_2({util::slice(2, -1, -1),
                                         util::slice(1, 2),
                                         util::slice(2, 0, -2)});
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
    REGISTER_TYPED_TEST_CASE_P(TestNdArrayIndex,
                               TestOperatorSquareBracket,
                               TestOperatorParenthesis1,
                               TestOperatorParenthesis2);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayIndex, MyIntTypes);

    template <typename T>
    class TestNdArrayFilter : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayFilter);

    TYPED_TEST_P(TestNdArrayFilter, TestWhere) {
        // shape(mask) == shape(array)
       {ndarray<bool> mask = (r_(std::vector<bool>({true, false, true, false}))
                              .reshape(shape_t({2, 2})));
        ndarray<TypeParam> x = (arange<int>(0, mask.size(), 1)
                                .reshape(shape_t({2, 2}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = x.where(mask);
        ASSERT_EQ(y.shape(), shape_t({2,}));
        ASSERT_EQ(y.size(), size_t(2));
        for(size_t i = 0; i < 2; ++i) {
            TypeParam const& tested_elem = y[{i}];
            TypeParam const& correct_elem = x[{i, 0}];
            ASSERT_EQ(tested_elem, correct_elem);
        }}

        // shape(mask) broadcasts
       {ndarray<bool> mask = nd::r_(std::vector<bool>({true, false}));
       ndarray<TypeParam> x = (arange<int>(0, 4, 1)
                               .reshape(shape_t({2, 2}))
                               .template cast<TypeParam>());

        ndarray<TypeParam> y = x.where(mask);
        ASSERT_EQ(y.shape(), shape_t({2,}));
        ASSERT_EQ(y.size(), size_t(2));
        for(size_t i = 0; i < 2; ++i) {
            TypeParam const& tested_elem = y[{i}];
            TypeParam const& correct_elem = x[{i, 0}];
            ASSERT_EQ(tested_elem, correct_elem);
        }}
    }

    TYPED_TEST_P(TestNdArrayFilter, TestFilter) {
        // shape(mask) == shape(array), scalar x
       {ndarray<bool> mask = (r_(std::vector<bool>({true, false, true, false}))
                              .reshape(shape_t({2, 2})));
        ndarray<TypeParam> ar = (arange<int>(0, mask.size(), 1)
                                 .reshape(shape_t({2, 2}))
                                 .template cast<TypeParam>());
        TypeParam const x = static_cast<TypeParam>(1.0);

        auto const& ar2 = ar.filter(mask, x);
        ASSERT_TRUE(ar2.equals(ar));

        {TypeParam const& tested_elem = ar2[{0, 0}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{0, 1}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{1, 0}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{1, 1}];
         TypeParam const correct_elem = static_cast<TypeParam>(3.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        }

        // shape(mask) == shape(array), vector x
       {ndarray<bool> mask = (r_(std::vector<bool>({true, false, true, false}))
                              .reshape(shape_t({2, 2})));
        ndarray<TypeParam> ar = (arange<int>(0, mask.size(), 1)
                                 .reshape(shape_t({2, 2}))
                                 .template cast<TypeParam>());
        ndarray<TypeParam> x = ((ndarray<int>(2) * arange<int>(0, 2, 1) + ndarray<int>(1))
                                .template cast<TypeParam>());

        auto const& ar2 = ar.filter(mask, x);
        ASSERT_TRUE(ar2.equals(ar));

        {TypeParam const& tested_elem = ar2[{0, 0}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{0, 1}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{1, 0}];
         TypeParam const correct_elem = static_cast<TypeParam>(3.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{1, 1}];
         TypeParam const correct_elem = static_cast<TypeParam>(3.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        }

        // shape(mask) broadcasts, scalar x
       {ndarray<bool> mask = r_(std::vector<bool>({true, false}));
        ndarray<TypeParam> ar = (arange<int>(0, 4, 1)
                                 .reshape(shape_t({2, 2}))
                                 .template cast<TypeParam>());
        TypeParam const x = static_cast<TypeParam>(1.0);

        auto const& ar2 = ar.filter(mask, x);
        ASSERT_TRUE(ar2.equals(ar));

        {TypeParam const& tested_elem = ar2[{0, 0}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{0, 1}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{1, 0}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{1, 1}];
         TypeParam const correct_elem = static_cast<TypeParam>(3.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        }

        // shape(mask) broadcasts, vector x
       {ndarray<bool> mask = r_(std::vector<bool>({true, false}));
        ndarray<TypeParam> ar = (arange<int>(0, 4, 1)
                                 .reshape(shape_t({2, 2}))
                                 .template cast<TypeParam>());
        ndarray<TypeParam> x = ((ndarray<int>(2) * arange<int>(0, 2, 1) + ndarray<int>(1))
                                .template cast<TypeParam>());

        auto const& ar2 = ar.filter(mask, x);
        ASSERT_TRUE(ar2.equals(ar));

        {TypeParam const& tested_elem = ar2[{0, 0}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{0, 1}];
         TypeParam const correct_elem = static_cast<TypeParam>(1.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{1, 0}];
         TypeParam const correct_elem = static_cast<TypeParam>(3.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        {TypeParam const& tested_elem = ar2[{1, 1}];
         TypeParam const correct_elem = static_cast<TypeParam>(3.0);
         ASSERT_EQ(tested_elem, correct_elem);
         }
        }
    }

    REGISTER_TYPED_TEST_CASE_P(TestNdArrayFilter,
                               TestWhere,
                               TestFilter);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayFilter, MyArithmeticTypes);

    template <typename T>
    class TestNdArrayIterate : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayIterate);
    TYPED_TEST_P(TestNdArrayIterate, TestNdArrayBeginEnd) {
        ndarray<TypeParam> x = (arange<int>(0, 5 * 3 * 4, 1)
                                .reshape(shape_t({5, 3, 4}))
                                .template cast<TypeParam>());

        size_t i = 0;
        for(auto it = x.begin(); it != x.end(); ++it, ++i) {
            TypeParam const& tested_elem = *it;
            TypeParam const& correct_elem = x.data()[i];
            ASSERT_EQ(tested_elem, correct_elem);
        }

        ndarray<TypeParam> y = x({util::slice(5, -1, -2)});
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

    REGISTER_TYPED_TEST_CASE_P(TestNdArrayIterate,
                               TestNdArrayBeginEnd);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayIterate, MyArithmeticTypes);



    /* Manipulation ======================================================== */
    template <typename T>
    class TestNdArrayManipulation : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayManipulation);

    TYPED_TEST_P(TestNdArrayManipulation, TestCopy) {
        // Contiguous array
        ndarray<TypeParam> x = (arange<int>(0, 20 * 30 * 40, 1)
                                .reshape(shape_t({20, 30, 40}))
                                .template cast<TypeParam>());

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
        std::vector<TypeParam> correct_y({static_cast<TypeParam>(22810),
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
                                          static_cast<TypeParam>( 4964)});
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
       {ndarray<TypeParam> x = (arange<int>(0, 10 * 10 * 5, 1)
                                . template cast<TypeParam>());
        ndarray<TypeParam> y = x.reshape(shape_t({10, 10, 5}));
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
        ndarray<TypeParam> y = x.reshape(shape_t({5, 10, 60}));
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
       {ndarray<TypeParam> x = (arange<int>(0, 10 * 20 * 30, 1)
                                .reshape(shape_t({10, 20, 30}))
                                .template cast<TypeParam>());

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
       {ndarray<TypeParam> x = (arange<int>(0, 2 * 3 * 4, 1)
                                .reshape(shape_t({2, 3, 4}))
                                .template cast<TypeParam>());

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

       {ndarray<TypeParam> x = (arange<int>(0, 2 * 1 * 4, 1)
                                .reshape(shape_t({2, 1, 4}))
                                .template cast<TypeParam>());

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

       {ndarray<TypeParam> x = (arange<int>(0, 2 * 1 * 4, 1)
                                .reshape(shape_t({2, 1, 4}))
                                .template cast<TypeParam>());

        ndarray<TypeParam> y = x.broadcast_to(shape_t({2, 1, 4}));
        ASSERT_TRUE(x.equals(y));}
    }

    REGISTER_TYPED_TEST_CASE_P(TestNdArrayManipulation,
                               TestCopy,
                               TestAsContiguousArray,
                               TestSqueeze,
                               TestReshape,
                               TestRavel,
                               TestBroadcastTo);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayManipulation, MyIntFloatComplexTypes);

    template <typename T>
    class TestNdArrayCast : public ::testing::Test {};
    TYPED_TEST_CASE_P(TestNdArrayCast);

    TYPED_TEST_P(TestNdArrayCast, TestCast) {
        using T = float;

        ndarray<TypeParam> x = zeros<TypeParam>({5, 3, 4});
        ndarray<T> y = x.template cast<T>();

        auto it_x = x.begin();
        for(auto it_y = y.begin(); it_y != y.end(); ++it_x, ++it_y) {
            TypeParam const& correct_elem = static_cast<T>(*it_x);
            T         const& tested_elem  = *it_y;
            ASSERT_EQ(tested_elem, correct_elem);
        }
    }

    REGISTER_TYPED_TEST_CASE_P(TestNdArrayCast,
                               TestCast);
    INSTANTIATE_TYPED_TEST_CASE_P(My, TestNdArrayCast, MyIntFloatTypes);
}

#endif // TEST_NDARRAY_CPP
