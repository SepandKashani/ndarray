// ############################################################################
// _ndlinalg.hpp
// =============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDLINALG_HPP
#define _NDLINALG_HPP

#include <algorithm>
#include <sstream>
#include <vector>

#include "_ndforward.hpp"

namespace nd::linalg {
    template <typename T>
    ndarray<T> mm(ndarray<T> const& A,
                  ndarray<T> const& B,
                  ndarray<T>* const out) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        shape_t const& sh_A = A.shape();
        shape_t const& sh_B = B.shape();
        shape_t sh_C(std::max<size_t>(A.ndim() + B.ndim() - 2, 1), 1);
        std::copy_n(sh_A.begin(), A.ndim() - 1, sh_C.begin());
        std::copy_n(sh_B.rbegin(), B.ndim() - 1, sh_C.rbegin());

        { // Validate parameters
           {bool const correct_shape = sh_A[A.ndim() - 1] == sh_B[0];
            std::stringstream error_msg;
            error_msg << "Cannon multiply arrays of shape {"
                      << sh_A << ", " << sh_B << "}. \n";
            util::NDARRAY_ASSERT(correct_shape, error_msg.str());}

            if(out != nullptr) {
                std::stringstream error_msg;
                error_msg << "Parameter[out]: Expected " << sh_C << " array, "
                          << "got " << out->shape() << ".\n";
                util::NDARRAY_ASSERT(sh_C == out->shape(), error_msg.str());
                util::NDARRAY_ASSERT(out->is_contiguous(),
                                     "Parameter[out] must point to a contiguous array.");
            }
        }

        /*
         * TODO: [A,B].reshape() forces a copy if arrays are non-contiguous,
         * but this is not mandatory to do matmul() through Eigen.
         *
         * Possible solution to get (A2, B2)::
         *
         *     if is_eigen_mappable([A,B]) && 2d:
         *         A2 = A
         *         B2 = B
         *     elif is_eigen_mappable([A,B]) && 1d:
         *         A2 = A(A.data(), {1, sh_A2_c}, {sizeof(T), A.strides()[0]})
         *         B2 = B(B.data(), {sh_B2_r, 1}, {B.strides()[0], sizeof(T)})
         *     else:
         *         A2 = A.reshape({sh_A2_r, sh_A2_c})
         *         B2 = B.reshape({sh_B2_r, sh_B2_c})
         */
        size_t const sh_A2_r = std::accumulate(sh_A.begin(), sh_A.end() - 1, size_t(1), std::multiplies<size_t>());
        size_t const sh_A2_c = sh_A[A.ndim() - 1];
        size_t const sh_B2_r = sh_B[0];
        size_t const sh_B2_c = std::accumulate(sh_B.begin() + 1, sh_B.end(), size_t(1), std::multiplies<size_t>());

        auto A2 = A.reshape({sh_A2_r, sh_A2_c});
        auto B2 = B.reshape({sh_B2_r, sh_B2_c});
        auto C = (out == nullptr) ? ndarray<T>(sh_C) : *out;
        auto C2 = C.reshape({sh_A2_r, sh_B2_c});

        auto const eA2 = util::interop::aseigenarray<T, mapM_t<T>>(&A2);
        auto const eB2 = util::interop::aseigenarray<T, mapM_t<T>>(&B2);
        auto       eC2 = util::interop::aseigenarray<T, mapM_t<T>>(&C2);
        (*eC2) = (*eA2) * (*eB2);  // C now contains the correct result.
        return C;
    }

    template <typename T>
    ndarray<T> bmm(ndarray<T> const& A,
                   ndarray<T> const& B,
                   ndarray<T>* const out) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        util::NDARRAY_ASSERT((A.ndim() == 2) || (A.ndim() == 3), "Parameter[A] must be 2D or 3D.");
        util::NDARRAY_ASSERT((B.ndim() == 2) || (B.ndim() == 3), "Parameter[B] must be 2D or 3D.");

        /*
         * TODO: .reshape() will cause a copy if (A, B) are not contiguous.
         * Call ndarray<T>(byte_t* const, shape_t const&, stride_t const&) instead.
         */
        auto const& AA = (A.ndim() == 3) ? A : A.reshape({1, A.shape()[0], A.shape()[1]});
        auto const& BB = (B.ndim() == 3) ? B : B.reshape({1, B.shape()[0], B.shape()[1]});

        shape_t const& sh_AA = AA.shape();
        shape_t const& sh_BB = BB.shape();

        { // Validate shapes
            bool const correct_shape = (((sh_AA[0] == sh_BB[0]) ||
                                         (sh_AA[0] == 1) ||
                                         (sh_BB[0] == 1)) &&
                                        (sh_AA[2] == sh_BB[1]));
            std::stringstream error_msg;
            error_msg << "Cannon broadcast-multiply arrays of shape {"
                      << A.shape() << ", " << B.shape() << "}. \n";
            util::NDARRAY_ASSERT(correct_shape, error_msg.str());
        }

        size_t const M(std::max(sh_AA[0], sh_BB[0]));
        size_t const N(sh_AA[1]);
        size_t const P(sh_BB[1]);
        size_t const Q(sh_BB[2]);
        auto const& bcast_AA = AA.broadcast_to({M, N, P});
        auto const& bcast_BB = BB.broadcast_to({M, P, Q});
        shape_t const sh_C({M, N, Q});
        if(out != nullptr) {
            std::stringstream error_msg;
            error_msg << "Parameter[out]: Expected " << sh_C << " array, "
                      << "got " << out->shape() << ".\n";
            util::NDARRAY_ASSERT(sh_C == out->shape(), error_msg.str());
            util::NDARRAY_ASSERT(out->is_contiguous(), "Parameter[out] must point to a contiguous array.");
        }
        ndarray<T> C = (out == nullptr) ? ndarray<T>(sh_C) : *out;

        auto select = std::vector(3, util::slice());
        for(size_t i = 0; i < M; ++i) {
            select[0] = util::slice(i, i + 1);

            /*
             * TODO: .reshape() will cause a copy if (A, B) are not contiguous.
             * Call ndarray<T>(byte_t* const, shape_t const&, stride_t const&) instead.
             */
            auto const& _A = bcast_AA(select).reshape({N, P});
            auto const& _B = bcast_BB(select).reshape({P, Q});
            auto _C = C(select).reshape({N, Q});

            mm(_A, _B, &_C);
        }

        return C;
    }
}

#endif // _NDLINALG_HPP
