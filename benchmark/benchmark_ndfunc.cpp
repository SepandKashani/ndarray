// ############################################################################
// benchmark_ndfunc.cpp
// ====================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef BENCHMARK_NDFUNC_CPP
#define BENCHMARK_NDFUNC_CPP

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

void BM_ARANGE(benchmark::State& state) {
    using T = double;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::arange<T>(0, 1024, 1));
    }
}

void BM_LINSPACE(benchmark::State& state) {
    using T = double;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::linspace<T>(0, 1, 1024));
    }
}

void BM_FULL(benchmark::State& state) {
    using T = nd::cdouble;

    T const value(1, 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::full<T>({1024,}, value));
    }
}

void BM_EYE(benchmark::State& state) {
    using T = nd::cdouble;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::eye<T>(32));
    }
}

void BM_ANY(benchmark::State& state) {
    nd::ndarray<bool> x({8, 32, 4});
    size_t axis = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::any(x, axis));
    }
}

void BM_ALL(benchmark::State& state) {
    nd::ndarray<bool> x({8, 32, 4});
    size_t axis = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::all(x, axis));
    }
}

void BM_ISCLOSE(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto y = x.copy();

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::isclose(x, y));
    }
}

void BM_ALLCLOSE(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto y = x.copy();

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::allclose(x, y));
    }
}

void BM_SUM(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    size_t axis = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::sum(x, axis));
    }
}

void BM_PROD(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    size_t axis = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::prod(x, axis));
    }
}

void BM_STACK(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto y = x + T(1);
    size_t axis = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::stack<T>({x, y}, axis));
    }
}

void BM_SIN(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-nd::pi<T>(), nd::pi<T>(), 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::sin(x));
    }
}

void BM_COS(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-nd::pi<T>(), nd::pi<T>(), 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::cos(x));
    }
}

void BM_TAN(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-nd::pi<T>(), nd::pi<T>(), 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::tan(x));
    }
}

void BM_ARCSIN(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::arcsin(x));
    }
}

void BM_ARCCOS(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::arccos(x));
    }
}

void BM_ARCTAN(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::arctan(x));
    }
}

void BM_ARCTAN2(benchmark::State& state) {
    using T = double;

    auto x1 = nd::linspace<T>(-1, 1, 1024);
    auto x2 = x1.copy();

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::arctan2(x1, x2));
    }
}

void BM_DEG2RAD(benchmark::State& state) {
    using T = double;

    auto deg = nd::linspace<T>(0, 360, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::deg2rad(deg));
    }
}

void BM_RAD2DEG(benchmark::State& state) {
    using T = double;

    auto rad = nd::linspace<T>(0, 2 * nd::pi<T>(), 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::rad2deg(rad));
    }
}

void BM_SINC(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::sinc(x));
    }
}

void BM_ABS(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::linspace<double>(-1, 1, 1024)
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::abs(x));
    }
}

void BM_EXP(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::exp(x));
    }
}

void BM_LOG(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(0, 2, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::log(x));
    }
}

void BM_UNIQUE(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::unique(x));
    }
}

void BM_STD(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    size_t axis = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::std(x, axis));
    }
}

void BM_MEAN(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    size_t axis = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::mean(x, axis));
    }
}

void BM_MIN(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    size_t axis = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::min(x, axis));
    }
}

void BM_MAX(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    size_t axis = 0;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::max(x, axis));
    }
}

void BM_CEIL(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::ceil(x));
    }
}

void BM_FLOOR(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::floor(x));
    }
}

void BM_CLIP(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);
    T down = 0;
    T up = 0.5;

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::clip(x, down, up));
    }
}

void BM_SIGN(benchmark::State& state) {
    using T = double;

    auto x = nd::linspace<T>(-1, 1, 1024);

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::sign(x));
    }
}

void BM_ASFLOAT(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::linspace<double>(-1, 1, 1024)
              .reshape({8, 32, 4})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::asfloat(x));
    }
}

void BM_REAL(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::linspace<double>(-1, 1, 1024)
              .reshape({8, 32, 4})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::real(x));
    }
}

void BM_IMAG(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::linspace<double>(-1, 1, 1024)
              .reshape({8, 32, 4})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::imag(x));
    }
}

void BM_CONJ(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::linspace<double>(-1, 1, 1024)
              .reshape({8, 32, 4})
              .cast<T>());

    for (auto _ : state) {
        benchmark::DoNotOptimize(nd::conj(x));
    }
}



BENCHMARK(BM_ARANGE);
BENCHMARK(BM_LINSPACE);
BENCHMARK(BM_FULL);
BENCHMARK(BM_EYE);
BENCHMARK(BM_ANY);
BENCHMARK(BM_ALL);
BENCHMARK(BM_ISCLOSE);
BENCHMARK(BM_ALLCLOSE);
BENCHMARK(BM_SUM);
BENCHMARK(BM_PROD);
BENCHMARK(BM_STACK);
BENCHMARK(BM_SIN);
BENCHMARK(BM_COS);
BENCHMARK(BM_TAN);
BENCHMARK(BM_ARCSIN);
BENCHMARK(BM_ARCCOS);
BENCHMARK(BM_ARCTAN);
BENCHMARK(BM_ARCTAN2);
BENCHMARK(BM_DEG2RAD);
BENCHMARK(BM_RAD2DEG);
BENCHMARK(BM_SINC);
BENCHMARK(BM_ABS);
BENCHMARK(BM_EXP);
BENCHMARK(BM_LOG);
BENCHMARK(BM_UNIQUE);
BENCHMARK(BM_STD);
BENCHMARK(BM_MEAN);
BENCHMARK(BM_MIN);
BENCHMARK(BM_MAX);
BENCHMARK(BM_CEIL);
BENCHMARK(BM_FLOOR);
BENCHMARK(BM_CLIP);
BENCHMARK(BM_SIGN);
BENCHMARK(BM_ASFLOAT);
BENCHMARK(BM_REAL);
BENCHMARK(BM_IMAG);
BENCHMARK(BM_CONJ);

#endif // BENCHMARK_NDFUNC_CPP
