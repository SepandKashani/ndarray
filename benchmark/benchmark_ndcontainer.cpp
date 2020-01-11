// ############################################################################
// benchmark_ndcontainer.cpp
// =========================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef BENCHMARK_NDCONTAINER_CPP
#define BENCHMARK_NDCONTAINER_CPP

#include <benchmark/benchmark.h>

#include "ndarray/ndarray.hpp"

void BM_NDCONTAINER_CONSTRUCTOR_SIZED(benchmark::State& state, size_t const nbytes) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::ndcontainer(nbytes));
    }
}
BENCHMARK_CAPTURE(BM_NDCONTAINER_CONSTRUCTOR_SIZED, cd_2p00, sizeof(nd::cdouble));
BENCHMARK_CAPTURE(BM_NDCONTAINER_CONSTRUCTOR_SIZED, cd_2p10, sizeof(nd::cdouble) * (1 << 10));
BENCHMARK_CAPTURE(BM_NDCONTAINER_CONSTRUCTOR_SIZED, cd_2p20, sizeof(nd::cdouble) * (1 << 20));
BENCHMARK_CAPTURE(BM_NDCONTAINER_CONSTRUCTOR_SIZED, cd_2p30, sizeof(nd::cdouble) * (1 << 30));

#endif // BENCHMARK_NDCONTAINER_CPP
