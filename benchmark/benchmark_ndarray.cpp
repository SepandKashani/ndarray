// ############################################################################
// benchmark_ndarray.cpp
// =====================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef BENCHMARK_NDARRAY_CPP
#define BENCHMARK_NDARRAY_CPP

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
 *     * If integer-valued types are supported -> test against int.
 */

void BM_NDARRAY_CONSTRUCTOR_SHAPE(benchmark::State& state,
                                  nd::shape_t sh) {
    using T = nd::cdouble;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::ndarray<T>(sh));
    }
}

void BM_NDARRAY_CONSTRUCTOR_COPY(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({1024,});

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::ndarray<T>(x));
    }
}

void BM_NDARRAY_EQUALS(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({1024,});
    nd::ndarray<T> y(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.equals(y));
    }
}

void BM_NDARRAY_AT(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({1024,});
    nd::index_t idx({5,});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.at(idx));
    }
}

void BM_NDARRAY_OP_SQUARE_BRACKET(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({1024,});
    nd::index_t idx({5,});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x[idx]);
    }
}

void BM_NDARRAY_OP_PARENTHESIS(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({8, 32, 4});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x({{0, 8, 2},
                                    {3, 16, 3}}));
    }
}

void BM_NDARRAY_WHERE(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({8, 32, 4});
    auto mask = nd::r_<bool>({false, true, true, false});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.where(mask));
    }
}

void BM_NDARRAY_FILTER_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> A({8, 32, 4});
    auto mask = nd::r_<bool>({false, true, true, false});
    auto x = ((nd::arange<int>(0, 8 * 32 * 2, 1) + 2.5)
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(A.filter(mask, x));
    }
}

void BM_NDARRAY_FILTER_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> A({8, 32, 4});
    auto mask = nd::r_<bool>({false, true, true, false});
    T x = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(A.filter(mask, x));
    }
}

void BM_NDARRAY_COPY(benchmark::State& state,
                     nd::shape_t sh) {
    using T = nd::cdouble;

    nd::ndarray<T> x(sh);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.copy());
    }
}

void BM_NDARRAY_SQUEEZE(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({1, 8, 1, 32, 4, 1});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.squeeze());
    }
}

void BM_NDARRAY_RESHAPE(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({32, 32});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.reshape({32 * 32,}));
    }
}

void BM_NDARRAY_RAVEL_CONTIGUOUS(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({32, 32});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.ravel());
    }
}

void BM_NDARRAY_RAVEL_STRIDED(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({8, 32, 4, 2});
    auto y = x({{}, {}, {}, {0, 1, 1}});

    for (auto _ : state) {
        benchmark::DoNotOptimize(y.ravel());
    }
}

void BM_NDARRAY_BROADCAST_TO(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({1024,});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.broadcast_to({256, 1, 1024}));
    }
}

void BM_NDARRAY_TRANSPOSE(benchmark::State& state) {
    using T = nd::cdouble;

    nd::ndarray<T> x({1, 8, 32, 1, 4});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.transpose());
    }
}

void BM_NDARRAY_CAST(benchmark::State& state) {
    using T = nd::cdouble;
    using T_c = nd::cfloat;

    nd::ndarray<T> x({1024,});

    for (auto _ : state) {
        benchmark::DoNotOptimize(x.cast<T_c>());
    }
}



BENCHMARK_CAPTURE(BM_NDARRAY_CONSTRUCTOR_SHAPE, cd_2p10, {(1 << 10),});
BENCHMARK_CAPTURE(BM_NDARRAY_CONSTRUCTOR_SHAPE, cd_2p20, {(1 << 20),});
BENCHMARK_CAPTURE(BM_NDARRAY_CONSTRUCTOR_SHAPE, cd_2p30, {(1 << 30),});
BENCHMARK(BM_NDARRAY_CONSTRUCTOR_COPY);
BENCHMARK(BM_NDARRAY_EQUALS);
BENCHMARK(BM_NDARRAY_AT);
BENCHMARK(BM_NDARRAY_OP_SQUARE_BRACKET);
BENCHMARK(BM_NDARRAY_OP_PARENTHESIS);
BENCHMARK(BM_NDARRAY_WHERE);
BENCHMARK(BM_NDARRAY_FILTER_NDARRAY);
BENCHMARK(BM_NDARRAY_FILTER_SCALAR);
BENCHMARK_CAPTURE(BM_NDARRAY_COPY, cd_2p10, {(1 << 10),});
BENCHMARK_CAPTURE(BM_NDARRAY_COPY, cd_2p20, {(1 << 20),});
BENCHMARK(BM_NDARRAY_SQUEEZE);
BENCHMARK(BM_NDARRAY_RESHAPE);
BENCHMARK(BM_NDARRAY_RAVEL_CONTIGUOUS);
BENCHMARK(BM_NDARRAY_RAVEL_STRIDED);
BENCHMARK(BM_NDARRAY_BROADCAST_TO);
BENCHMARK(BM_NDARRAY_TRANSPOSE);
BENCHMARK(BM_NDARRAY_CAST);

#endif // BENCHMARK_NDARRAY_CPP
