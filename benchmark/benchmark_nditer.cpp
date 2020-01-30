// ############################################################################
// benchmark_nditer.cpp
// ====================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef BENCHMARK_NDITER_CPP
#define BENCHMARK_NDITER_CPP

#include <benchmark/benchmark.h>
#include "benchmark_type.hpp"

#include "ndarray/ndarray.hpp"

/* Note
 * ----
 * Benchmarks of the form `void f(benchmark::State&)` all operate on arrays of
 * size 1024 as a baseline.
 *
 * The goal is to assess the speed at which an array can be traversed (and
 * dereferenced) based on its shape/strides.
 */

void BM_NDITER_1D(benchmark::State& state) {
    using T = int;

    auto x = nd::arange<T>(0, 1024, 1);

    for (auto _ : state) {
        for (auto _x = x.begin(); _x != x.end(); ++_x) {}
    }
}

void BM_NDITER_1D_WITH_DEREFERENCE(benchmark::State& state) {
    using T = int;

    auto x = nd::arange<T>(0, 1024, 1);

    for (auto _ : state) {
        for (auto _x = x.begin(); _x != x.end(); ++_x) {
            benchmark::DoNotOptimize(*_x);
        }
    }
}

void BM_NDITER_1D_MANUAL_WITH_DEREFERENCE(benchmark::State& state) {
    /*
     * Special case where we use the raw pointers underlying ndarrays to perform
     * the iteration. This should give a lower bound on the achievable iteration
     * speed.
     */
    using T = int;

    auto x = nd::arange<T>(0, 1024, 1);

    for (auto _ : state) {
        for (auto _x = x.data(); _x != x.data() + x.size(); ++_x) {
            benchmark::DoNotOptimize(*_x);
        }
    }
}

void BM_NDITER_1D_EFFECTIVE(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({1, 1024, 1}));

    for (auto _ : state) {
        for (auto _x = x.begin(); _x != x.end(); ++_x) {}
    }
}

void BM_NDITER_1D_EFFECTIVE_WITH_DEREFERENCE(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({1, 1024, 1}));

    for (auto _ : state) {
        for (auto _x = x.begin(); _x != x.end(); ++_x) {
            benchmark::DoNotOptimize(*_x);
        }
    }
}

void BM_NDITER_ND_CONTIGUOUS(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({32, 8, 4}));

    for (auto _ : state) {
        for (auto _x = x.begin(); _x != x.end(); ++_x) {}
    }
}

void BM_NDITER_ND_CONTIGUOUS_WITH_DEREFERENCE(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({32, 8, 4}));

    for (auto _ : state) {
        for (auto _x = x.begin(); _x != x.end(); ++_x) {
            benchmark::DoNotOptimize(*_x);
        }
    }
}

void BM_NDITER_ND_STRIDED(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024 * 2 * 3 * 4, 1)
              .reshape({32 * 2, 8 * 3, 4 * 4})
              .operator()({{0, 32 * 2, 2},
                           {0,  8 * 3, 3},
                           {0,  4 * 4, 4}}));

    for (auto _ : state) {
        for (auto _x = x.begin(); _x != x.end(); ++_x) {}
    }
}

void BM_NDITER_ND_STRIDED_WITH_DEREFERENCE(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024 * 2 * 3 * 4, 1)
              .reshape({32 * 2, 8 * 3, 4 * 4})
              .operator()({{0, 32 * 2, 2},
                           {0,  8 * 3, 3},
                           {0,  4 * 4, 4}}));

    for (auto _ : state) {
        for (auto _x = x.begin(); _x != x.end(); ++_x) {
            benchmark::DoNotOptimize(*_x);
        }
    }
}



BENCHMARK(BM_NDITER_1D);
BENCHMARK(BM_NDITER_1D_WITH_DEREFERENCE);
BENCHMARK(BM_NDITER_1D_MANUAL_WITH_DEREFERENCE);
BENCHMARK(BM_NDITER_1D_EFFECTIVE);
BENCHMARK(BM_NDITER_1D_EFFECTIVE_WITH_DEREFERENCE);
BENCHMARK(BM_NDITER_ND_CONTIGUOUS);
BENCHMARK(BM_NDITER_ND_CONTIGUOUS_WITH_DEREFERENCE);
BENCHMARK(BM_NDITER_ND_STRIDED);
BENCHMARK(BM_NDITER_ND_STRIDED_WITH_DEREFERENCE);

void BM_NDITER_ADVANCE(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024 * 2 * 3 * 4, 1)
              .reshape({32 * 2, 8 * 3, 4 * 4}));
    size_t N = x.size();

    for (auto _ : state) {
        auto _x = x.begin();
        for (size_t i = 0; i < N; ++i, ++_x) {}
    }
}
void BM_NDITER_ADVANCE_MANUAL(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024 * 2 * 3 * 4, 1)
              .reshape({32 * 2, 8 * 3, 4 * 4}));
    size_t N = x.size();

    for (auto _ : state) {
        auto _x = x.data();
        for (size_t i = 0; i < N; ++i, ++_x) {}
    }
}
BENCHMARK(BM_NDITER_ADVANCE);
BENCHMARK(BM_NDITER_ADVANCE_MANUAL);

#endif // BENCHMARK_NDITER_CPP
