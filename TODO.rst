.. ############################################################################
.. TODO.rst
.. ========
.. Author : Sepand KASHANI [kashani.sepand@gmail.com]
.. ############################################################################


nd::util::apply() interface

    * Do not assume `out` is provided and allocate if required. Return `out`
      instead of void.

    Advantage::

        * Simplifies all function definitions that use apply(): no need to have
          a separate code path for `out != nullptr`;

        * Simplifies testing code: no need to have separate test cases for `out
          != nullptr`. All these tests move into apply()'s tests.

nd::util::reduce() interface

    * Do not assume `out` is provided and allocate if required.
      Return `out` instead of void.

    Advantage::

        * Simplifies all function definitions that use reduce(): no need to have
          a separate code path for `out != nullptr`;

        * Simplifies testing code: no need to have separate test cases for
          `out != nullptr`. All these tests move into reduce()'s tests.

nd::util::[apply, reduce, reduce3D]() interface

    * Allow choice of execution policy.
      Investigate <execution> header for details.

nd::ndarray_iterator<T>

    It seems impossible to provide any meaningful speedups for the general
    multi-dim case. The best solution is to offer tailored polymorphic iteration
    based on properties of the array, but this will have a runtime cost itself.
    In the interest of speed, we will go for a concrete-class implementation as
    below.

    ndarray_iterator<T>
        private
            ndarray<T>* m_iterable
            index_t m_index
            int m_offset  // byte offset
            iter_tag = {single_stride, multi_stride}

            void advance() // implementation varies based on tag

        public
            ndarray_iterator() = delete;
            ndarray_iterator(ndarray<T>* const x);
            ndarray_iterator(ndarray<T>* const x, index_t const& index);
            ndarray_iterator(ndarray_iterator<T> const& other);
            ~ndarray_iterator();

            ndarray_iterator<T>& operator=(ndarray_iterator<T> const& other);
            bool operator==(ndarray_iterator<T> const& other) const;
            bool operator!=(ndarray_iterator<T> const& other) const;
            T& operator*() const;
            ndarray_iterator<T>& operator++();

    Will need to redefine what begin()/end() mean for these different iterators.
    Overhaul iterator tests to verify functionality.
    Do not break public interface of ndarray.
    Benchmark w.r.t WIP commit.

nd::ndarray<T>::where()

    * use std::inserter() to populate `buffer`.

    * This still does not solve the two-copy problem.

nd::ndarray<T>::filter()

    * if `broadcast_mode` : use std::replace_if()

    * if `!broadcast_mode`: change to a two-stage algorithm.

        * Assert mask_bcast.sum() == x.size();

        * check std::replace_if() implementation for inspiration. (for loop)

nd::ndarray<T>::filter(ndarray<bool> const& mask, T const& x)

    * Change interface to filter(ndarray<bool> const& mask, T const x).

    * Don't use `ndarray<T> _x({1})`.
      Replace with `ndarray<T> _x(reinterpret_cast<byte_t*>(&x), {1})`.

nd::stack()

    `x[i].reshape(sh_x)` will force a copy if not contiguous. Instead it is
    possible to omit the copy by inserting a phantom dimension at the right
    location. (This should be a method in ndarray<T> as it may be useful for
    other functions too)

nd::linalg::mm()

    [A,B].reshape() forces a copy if arrays are non-contiguous, but this is not
    mandatory to do matmul() through Eigen.

    Possible solution to get (A2, B2)::

        if is_eigen_mappable([A,B]) && 2d:
            A2 = A
            B2 = B
        elif is_eigen_mappable([A,B]) && 1d:
            A2 = A(A.data(), {1, sh_A2_c}, {sizeof(T), A.strides()[0]})
            B2 = B(B.data(), {sh_B2_r, 1}, {B.strides()[0], sizeof(T)})
        else:
            A2 = A.reshape({sh_A2_r, sh_A2_c})
            B2 = B.reshape({sh_B2_r, sh_B2_c})

nd::linalg::bmm()

    * Don't build as a shell around nd::linalg::mm() for efficiency.

    * High level procedure::

        assert(A.ndim() == {2, 3})
        assert(B.ndim() == {2, 3})

        sh_A_r, sh_A_c = A.shape[[-2, -1]]
        sh_B_r, sh_B_c = B.shape[[-2, -1]]
        assert(sh_A_c == sh_B_r);

        A3 = (A.ndim() == 2) ? A.broadcast_to({1, sh_A_r, sh_A_c}) : A
        B3 = (B.ndim() == 2) ? B.broadcast_to({1, sh_B_r, sh_B_c}) : B
        sh_A3_b, sh_B3_b = A3.shape[0], B3.shape[0]
        assert((sh_A3_b == sh_B3_b) or (sh_A3_b == 1) or (sh_B3_b == 1))

        sh_C3 = {max(sh_A3_b, sh_B3_b), sh_A_r, sh_B_c}
        if out != nullptr:
            assert((out.shape == sh_C3) and out->is_contiguous())
        C3 = (out == nullptr) ? ndarray<T>(sh_C3) : (*out)

        # Make sure input 2D slices are eigen-mappable, otherwise make them.
        # [AB]3m = mappable (sub-)arrays
        A3m = (eigen_mappable(A3({{0, 1}}).squeeze({0}) ? A3 : ascontiguousarray(A3)).broadcast_to({sh_C3[0], sh_A_r, sh_A_c})
        B3m = (eigen_mappable(B3({{0, 1}}).squeeze({0}) ? B3 : ascontiguousarray(B3)).broadcast_to({sh_C3[0], sh_B_r, sh_B_c})

        for i in range(sh_C3[0]):
            C3[i] = A3m[i] * B3m[i]  // suitably eigen-mapped

HTML DOCUMENTATION

SIMPLIFY TEST SUITE

    After introduction of new function interfaces

NEW_FUNCTIONS_TO_IMPLEMENT

    nd::ndarray<T>::expand_dims()
    nd::ndarray<T>::operator(nd::ndarray<bool> const& mask) instead of where()?
    nd::func::angle()
    nd::func::around()
    nd::func::concatenate()
    nd::func::exp10()
    nd::func::exp2()
    nd::func::load()
    nd::func::log10()
    nd::func::log2()
    nd::func::logspace()
    nd::func::minmax()
    nd::func::save()
    nd::func::sort()
    nd::func::view_as_windows()
    nd::linalg::eigh()
    nd::linalg::eigvalsh()
    nd::linalg::norm()
    nd::linalg::qr()
    nd::linalg::svd()
    nd::fft::czt()
    nd::fft::ffs()
    nd::fft::fft()
    nd::fft::fs_interp()
    nd::fft::fs_sample()
    nd::fft::iffs()
    nd::fft::ifft()
