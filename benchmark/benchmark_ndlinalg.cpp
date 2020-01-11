// ############################################################################
// benchmark_ndlinalg.cpp
// ======================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef BENCHMARK_NDLINALG_CPP
#define BENCHMARK_NDLINALG_CPP

#include <benchmark/benchmark.h>
#include "benchmark_type.hpp"

#include "ndarray/ndarray.hpp"

/* Note
 * ----
 * Benchmarks of the form `void f(benchmark::State&)` all operate on arrays of
 * size 1024 as a baseline.
 *
 *     * If complex-valued types are supported -> test against nd::cdouble;
 *     * If real-valued types are supported    -> test against double;
 */

void BM_LINALG_MM_2D_2D(benchmark::State& state) {
    using T = nd::cdouble;

    auto A = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 32})
              .cast<T>());
    auto B = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 32})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::linalg::mm(A, B));
    }
}

void BM_LINALG_MM_3D_2D(benchmark::State& state) {
    using T = nd::cdouble;

    auto A = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 4, 32})
              .cast<T>());
    auto B = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 32})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::linalg::mm(A, B));
    }
}

void BM_LINALG_MM_2D_3D(benchmark::State& state) {
    using T = nd::cdouble;

    auto A = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 32})
              .cast<T>());
    auto B = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 8, 4})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::linalg::mm(A, B));
    }
}

void BM_LINALG_MM_3D_3D(benchmark::State& state) {
    using T = nd::cdouble;

    auto A = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 4, 32})
              .cast<T>());
    auto B = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 8, 4})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::linalg::mm(A, B));
    }
}

void BM_LINALG_BMM_2D_2D(benchmark::State& state) {
    using T = nd::cdouble;

    auto A = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 32})
              .cast<T>());
    auto B = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 32})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::linalg::bmm(A, B));
    }
}

void BM_LINALG_BMM_3D_2D(benchmark::State& state) {
    using T = nd::cdouble;

    auto A = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 4, 32})
              .cast<T>());
    auto B = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 32})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::linalg::bmm(A, B));
    }
}

void BM_LINALG_BMM_2D_3D(benchmark::State& state) {
    using T = nd::cdouble;

    auto A = (nd::arange<int>(0, 1024, 1)
              .reshape({32, 32})
              .cast<T>());
    auto B = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::linalg::bmm(A, B));
    }
}

void BM_LINALG_BMM_3D_3D(benchmark::State& state) {
    using T = nd::cdouble;

    auto A = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 4, 32})
              .cast<T>());
    auto B = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::linalg::bmm(A, B));
    }
}



BENCHMARK(BM_LINALG_MM_2D_2D);
BENCHMARK(BM_LINALG_MM_3D_2D);
BENCHMARK(BM_LINALG_MM_2D_3D);
BENCHMARK(BM_LINALG_MM_3D_3D);
BENCHMARK(BM_LINALG_BMM_2D_2D);
BENCHMARK(BM_LINALG_BMM_3D_2D);
BENCHMARK(BM_LINALG_BMM_2D_3D);
BENCHMARK(BM_LINALG_BMM_3D_3D);

#endif // BENCHMARK_NDLINALG_CPP
