// ############################################################################
// benchmark_ndcontainer.cpp
// =========================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef BENCHMARK_NDCONTAINER_CPP
#define BENCHMARK_NDCONTAINER_CPP

#include <benchmark/benchmark.h>

#include "ndarray/ndarray.hpp"

static void ndcontainer_constructor_sized(benchmark::State& state) {
    for (auto _ : state) {
        size_t const nbytes = 1024u;
        nd::ndcontainer c(nbytes);
    }
}

BENCHMARK(ndcontainer_constructor_sized);

#endif // BENCHMARK_NDCONTAINER_CPP
