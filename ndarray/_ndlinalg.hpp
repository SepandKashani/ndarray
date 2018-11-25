// ############################################################################
// _ndlinalg.hpp
// =============
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef _NDLINALG_HPP
#define _NDLINALG_HPP

namespace nd { template <typename T> class ndarray; }

namespace nd { namespace linalg {
    /*
     * Matrix-Multiplication extended to ND-arrays.
     *
     * Parameters
     * ----------
     * A : ndarray<T> const&
     *     (a[0], ..., a[N-2], d) array.
     * B : ndarray<T> const&
     *     (d, b[0], ..., b[M-2]) array.
     *
     * Returns
     * -------
     * C : ndarray<T>
     *     (a[0], ..., a[N-2], b[0], ..., b[M-2]) array.
     *
     * Examples
     * --------
     * Let A \in \bR^{5 \times 3 \times 4} and B \in \bR^{4, 7, 2}.
     * Then C = mm(A, B) \in \bR^{5 \times 3 \times 7 \times 2} such that
     *     C[i,j,k,l] = \sum_{q = 0}^{3} A[i,j,q] B[q,k,l]
     */
    template <typename T>
    ndarray<T> mm(ndarray<T> const& A, ndarray<T> const& B, ndarray<T>* const out = nullptr) {
        return A + B;
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
}}

#endif // _NDLINALG_HPP
