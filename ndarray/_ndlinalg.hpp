// ############################################################################
// _ndlinalg.hpp
// =============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDLINALG_HPP
#define _NDLINALG_HPP

#include <algorithm>
#include <sstream>

#include "_ndarray.hpp"
#include "_ndtype.hpp"
#include "_ndutil.hpp"

namespace nd { template <typename T> class ndarray; }

namespace nd::linalg {
    /*
     * Matrix-Multiplication extended to ND-arrays.
     *
     * Parameters
     * ----------
     * A : ndarray<T> const&
     *     (a[0], ..., a[N-2], d) array.
     * B : ndarray<T> const&
     *     (d, b[1], ..., b[M-1]) array.
     * out : ndarray<T>* const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output, and be contiguous.
     *
     * Returns
     * -------
     * C : ndarray<T>
     *     (a[0], ..., a[N-2], b[1], ..., b[M-1]) array.
     *
     * Examples
     * --------
     * Let A \in \bR^{5, 3, 4} and B \in \bR^{4, 7, 2}.
     * Then C = mm(A, B) \in \bR^{5, 3, 7, 2} such that
     *     C[i,j,k,l] = \sum_{q = 0}^{3} A[i,j,q] B[q,k,l]
     */
    template <typename T>
    ndarray<T> mm(ndarray<T> const& A, ndarray<T> const& B, ndarray<T>* const out = nullptr) {
        static_assert(is_int<T>() || is_float<T>() || is_complex<T>(),
                      "Only {int, float, complex} types allowed.");

        shape_t const& sh_A = A.shape();
        shape_t const& sh_B = B.shape();

        { // Validate shapes
            bool const correct_shape = sh_A[A.ndim() - 1] == sh_B[0];
            std::stringstream error_msg;
            error_msg << "Cannon multiply arrays of shape {"
                      << sh_A << ", " << sh_B << "}. \n";
            util::NDARRAY_ASSERT(correct_shape, error_msg.str());
        }

        size_t const sh_Ar0 = std::accumulate(sh_A.begin(), sh_A.end() - 1, size_t(1), std::multiplies<size_t>());
        size_t const sh_Ar1 = sh_A[A.ndim() - 1];
        auto Ar = A.reshape({sh_Ar0, sh_Ar1});
        auto const eAr = util::interop::aseigenarray<T, mapM_t<T>>(&Ar);

        size_t const sh_Br0 = sh_B[0];
        size_t const sh_Br1 = std::accumulate(sh_B.begin() + 1, sh_B.end(), size_t(1), std::multiplies<size_t>());
        auto Br = B.reshape({sh_Br0, sh_Br1});
        auto const eBr = util::interop::aseigenarray<T, mapM_t<T>>(&Br);

        shape_t sh_C(std::max<size_t>(A.ndim() + B.ndim() - 2, 1), 1);
        std::copy_n(sh_A.cbegin(), A.ndim() - 1, sh_C.begin());
        std::copy_n(sh_B.crbegin(), B.ndim() - 1, sh_C.rbegin());
        if(out != nullptr) {
            std::stringstream error_msg;
            error_msg << "Parameter[out]: Expected " << sh_C << " array, "
                      << "got " << out->shape() << ".\n";
            util::NDARRAY_ASSERT(sh_C == out->shape(), error_msg.str());
            util::NDARRAY_ASSERT(out->is_contiguous(), "Parameter[out] must point to a contiguous array.");
        }
        ndarray<T> C = (out == nullptr) ? ndarray<T>(sh_C) : *out;
        auto Cr = C.reshape({sh_Ar0, sh_Br1});
        auto eCr = util::interop::aseigenarray<T, mapM_t<T>>(&Cr);

        (*eCr) = (*eAr) * (*eBr);
        return C;
    }

    /*
     * Batch Matrix-Multiplication.
     *
     * Parameters
     * ----------
     * A : ndarray<T> const&
     *     (N, M) or (..., N, M) array.
     *     If `A` is (N, M), it is transformed to a (1, N, M) array.
     * B : ndarray<T> const&
     *     (M, Q) or (..., M, Q) array.
     *     If `B` is (M, Q), it is transformed to a (1, M, Q) array.
     * out : ndarray<T> * const
     *     Optional buffer to store result.
     *     Must have the same dimensions as the output.
     *
     * Returns
     * -------
     * C : ndarray<T>
     *     (..., N, Q) layer-wise matrix product of `A` and `B`.
     *
     * Examples
     * --------
     * Let A \in \bR^{a1, d, N, M} and B \in \bR^{d, M, Q}.
     * Then C = bmm(A, B) \in \bR^{a1, d, N, Q} such that
     *     C[i, j, :, :] = mm(A[i, j, :, :], B[j, :, :])
     */
    template <typename T>
    ndarray<T> bmm(ndarray<T> const& A, ndarray<T> const& B, ndarray<T>* const out = nullptr);
}

#endif // _NDLINALG_HPP
