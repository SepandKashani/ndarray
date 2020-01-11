// ############################################################################
// benchmark_ndarray_operator.cpp
// ==============================
// Author : Sepand KASHANI [kashani.sepand@gmail.com]
// ############################################################################

#ifndef BENCHMARK_NDARRAY_OPERATOR_CPP
#define BENCHMARK_NDARRAY_OPERATOR_CPP

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

void BM_OPERATOR_EQUAL_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x = other);
    }
}

void BM_OPERATOR_EQUAL_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x = other);
    }
}

void BM_OPERATOR_PLUS_EQUAL_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x += other);
    }
}

void BM_OPERATOR_PLUS_EQUAL_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x += other);
    }
}

void BM_OPERATOR_MINUS_EQUAL_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x -= other);
    }
}

void BM_OPERATOR_MINUS_EQUAL_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x -= other);
    }
}

void BM_OPERATOR_TIMES_EQUAL_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x *= other);
    }
}

void BM_OPERATOR_TIMES_EQUAL_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x *= other);
    }
}

void BM_OPERATOR_DIVIDE_EQUAL_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto other = nd::ones<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x /= other);
    }
}

void BM_OPERATOR_DIVIDE_EQUAL_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x /= other);
    }
}

void BM_OPERATOR_MOD_EQUAL_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x %= other);
    }
}

void BM_OPERATOR_MOD_EQUAL_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x %= other);
    }
}

void BM_OPERATOR_AND_EQUAL_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x &= other);
    }
}

void BM_OPERATOR_AND_EQUAL_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x &= other);
    }
}

void BM_OPERATOR_OR_EQUAL_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x |= other);
    }
}

void BM_OPERATOR_OR_EQUAL_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x |= other);
    }
}

void BM_OPERATOR_XOR_EQUAL_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x ^= other);
    }
}

void BM_OPERATOR_XOR_EQUAL_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x ^= other);
    }
}

void BM_OPERATOR_SHIFT_LEFT_EQUAL_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x <<= other);
    }
}

void BM_OPERATOR_SHIFT_LEFT_EQUAL_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x <<= other);
    }
}

void BM_OPERATOR_SHIFT_RIGHT_EQUAL_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x >>= other);
    }
}

void BM_OPERATOR_SHIFT_RIGHT_EQUAL_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x >>= other);
    }
}

void BM_OPERATOR_PLUS_PLUS(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));

    for (auto _ : state) {
        benchmark::DoNotOptimize(++x);
    }
}

void BM_OPERATOR_MINUS_MINUS(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));

    for (auto _ : state) {
        benchmark::DoNotOptimize(--x);
    }
}

void BM_OPERATOR_UNARY_MINUS(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));

    for (auto _ : state) {
        benchmark::DoNotOptimize(-x);
    }
}

void BM_OPERATOR_UNARY_BIT_FLIP(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));

    for (auto _ : state) {
        benchmark::DoNotOptimize(~x);
    }
}

void BM_OPERATOR_NOT(benchmark::State& state) {
    nd::ndarray<bool> x({8, 32, 4});

    for (auto _ : state) {
        benchmark::DoNotOptimize(!x);
    }
}

void BM_OPERATOR_PLUS_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x + other);
    }
}

void BM_OPERATOR_PLUS_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x + other);
    }
}

void BM_OPERATOR_MINUS_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x - other);
    }
}

void BM_OPERATOR_MINUS_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x - other);
    }
}

void BM_OPERATOR_TIMES_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x * other);
    }
}

void BM_OPERATOR_TIMES_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x * other);
    }
}

void BM_OPERATOR_DIVIDE_NDARRAY(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    auto other = nd::ones<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x / other);
    }
}

void BM_OPERATOR_DIVIDE_SCALAR(benchmark::State& state) {
    using T = nd::cdouble;

    auto x = (nd::arange<int>(0, 1024, 1)
              .reshape({8, 32, 4})
              .cast<T>());
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x / other);
    }
}

void BM_OPERATOR_MOD_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x % other);
    }
}

void BM_OPERATOR_MOD_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x % other);
    }
}

void BM_OPERATOR_AND_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x & other);
    }
}

void BM_OPERATOR_AND_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x & other);
    }
}

void BM_OPERATOR_OR_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x | other);
    }
}

void BM_OPERATOR_OR_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x | other);
    }
}

void BM_OPERATOR_XOR_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x ^ other);
    }
}

void BM_OPERATOR_XOR_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x ^ other);
    }
}

void BM_OPERATOR_SHIFT_LEFT_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x << other);
    }
}

void BM_OPERATOR_SHIFT_LEFT_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x << other);
    }
}

void BM_OPERATOR_SHIFT_RIGHT_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::full<T>(x.shape(), 2);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x >> other);
    }
}

void BM_OPERATOR_SHIFT_RIGHT_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = 2;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x >> other);
    }
}

void BM_OPERATOR_AND_AND_NDARRAY(benchmark::State& state) {
    nd::ndarray<bool> x({8, 32, 4});
    nd::ndarray<bool> other(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x && other);
    }
}

void BM_OPERATOR_AND_AND_SCALAR(benchmark::State& state) {
    nd::ndarray<bool> x({8, 32, 4});
    bool other = true;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x && other);
    }
}

void BM_OPERATOR_OR_OR_NDARRAY(benchmark::State& state) {
    nd::ndarray<bool> x({8, 32, 4});
    nd::ndarray<bool> other(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x || other);
    }
}

void BM_OPERATOR_OR_OR_SCALAR(benchmark::State& state) {
    nd::ndarray<bool> x({8, 32, 4});
    bool other = false;

    for (auto _ : state) {
        benchmark::DoNotOptimize(x || other);
    }
}

void BM_OPERATOR_EQUAL_EQUAL_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x == other);
    }
}

void BM_OPERATOR_EQUAL_EQUAL_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x == other);
    }
}

void BM_OPERATOR_NOT_EQUAL_NDARRAY(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x != other);
    }
}

void BM_OPERATOR_NOT_EQUAL_SCALAR(benchmark::State& state) {
    using T = int;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x != other);
    }
}

void BM_OPERATOR_LESS_NDARRAY(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x < other);
    }
}

void BM_OPERATOR_LESS_SCALAR(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x < other);
    }
}

void BM_OPERATOR_LESS_EQUAL_NDARRAY(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x <= other);
    }
}

void BM_OPERATOR_LESS_EQUAL_SCALAR(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x <= other);
    }
}

void BM_OPERATOR_GREATER_NDARRAY(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x > other);
    }
}

void BM_OPERATOR_GREATER_SCALAR(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x > other);
    }
}

void BM_OPERATOR_GREATER_EQUAL_NDARRAY(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    auto other = nd::zeros<T>(x.shape());

    for (auto _ : state) {
        benchmark::DoNotOptimize(x >= other);
    }
}

void BM_OPERATOR_GREATER_EQUAL_SCALAR(benchmark::State& state) {
    using T = double;

    auto x = (nd::arange<T>(0, 1024, 1)
              .reshape({8, 32, 4}));
    T other = static_cast<T>(1);

    for (auto _ : state) {
        benchmark::DoNotOptimize(x >= other);
    }
}



BENCHMARK(BM_OPERATOR_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_PLUS_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_PLUS_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_MINUS_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_MINUS_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_TIMES_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_TIMES_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_DIVIDE_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_DIVIDE_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_MOD_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_MOD_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_AND_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_AND_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_OR_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_OR_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_XOR_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_XOR_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_SHIFT_LEFT_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_SHIFT_LEFT_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_SHIFT_RIGHT_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_SHIFT_RIGHT_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_PLUS_PLUS);
BENCHMARK(BM_OPERATOR_MINUS_MINUS);
BENCHMARK(BM_OPERATOR_UNARY_MINUS);
BENCHMARK(BM_OPERATOR_UNARY_BIT_FLIP);
BENCHMARK(BM_OPERATOR_NOT);
BENCHMARK(BM_OPERATOR_PLUS_NDARRAY);
BENCHMARK(BM_OPERATOR_PLUS_SCALAR);
BENCHMARK(BM_OPERATOR_MINUS_NDARRAY);
BENCHMARK(BM_OPERATOR_MINUS_SCALAR);
BENCHMARK(BM_OPERATOR_TIMES_NDARRAY);
BENCHMARK(BM_OPERATOR_TIMES_SCALAR);
BENCHMARK(BM_OPERATOR_DIVIDE_NDARRAY);
BENCHMARK(BM_OPERATOR_DIVIDE_SCALAR);
BENCHMARK(BM_OPERATOR_MOD_NDARRAY);
BENCHMARK(BM_OPERATOR_MOD_SCALAR);
BENCHMARK(BM_OPERATOR_AND_NDARRAY);
BENCHMARK(BM_OPERATOR_AND_SCALAR);
BENCHMARK(BM_OPERATOR_OR_NDARRAY);
BENCHMARK(BM_OPERATOR_OR_SCALAR);
BENCHMARK(BM_OPERATOR_XOR_NDARRAY);
BENCHMARK(BM_OPERATOR_XOR_SCALAR);
BENCHMARK(BM_OPERATOR_SHIFT_LEFT_NDARRAY);
BENCHMARK(BM_OPERATOR_SHIFT_LEFT_SCALAR);
BENCHMARK(BM_OPERATOR_SHIFT_RIGHT_NDARRAY);
BENCHMARK(BM_OPERATOR_SHIFT_RIGHT_SCALAR);
BENCHMARK(BM_OPERATOR_AND_AND_NDARRAY);
BENCHMARK(BM_OPERATOR_AND_AND_SCALAR);
BENCHMARK(BM_OPERATOR_OR_OR_NDARRAY);
BENCHMARK(BM_OPERATOR_OR_OR_SCALAR);
BENCHMARK(BM_OPERATOR_EQUAL_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_EQUAL_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_NOT_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_NOT_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_LESS_NDARRAY);
BENCHMARK(BM_OPERATOR_LESS_SCALAR);
BENCHMARK(BM_OPERATOR_LESS_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_LESS_EQUAL_SCALAR);
BENCHMARK(BM_OPERATOR_GREATER_NDARRAY);
BENCHMARK(BM_OPERATOR_GREATER_SCALAR);
BENCHMARK(BM_OPERATOR_GREATER_EQUAL_NDARRAY);
BENCHMARK(BM_OPERATOR_GREATER_EQUAL_SCALAR);

#endif // BENCHMARK_NDARRAY_OPERATOR_CPP
