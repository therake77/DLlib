#include <benchmark/benchmark.h>

#include <core/ConvLayer.hpp>
#include <core/FullyConnected.hpp>
#include <core/Tensor.hpp>
#include <hardware/DeviceManager.hpp>
#include <math/TensorMath.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

enum class BenchDevice {
    CPU,
    GPU
};

std::shared_ptr<Device> make_device(BenchDevice device) {
    if(device == BenchDevice::CPU) {
        return DeviceManager::get_cpu_device();
    }

#ifdef __USE_CUDA__
    return DeviceManager::get_gpu_device();
#else
    throw std::runtime_error("GPU benchmarks require CUDA support");
#endif
}

const char* device_name(BenchDevice device) {
    return device == BenchDevice::CPU ? "CPU" : "GPU";
}

void set_common_counters(
    benchmark::State& state,
    BenchDevice device,
    std::size_t elements
) {
    state.counters["elements"] = static_cast<double>(elements);
    state.counters["items_per_second"] = benchmark::Counter(
        static_cast<double>(elements),
        benchmark::Counter::kIsIterationInvariantRate
    );
    state.SetLabel(device_name(device));
}

void tensor_add(benchmark::State& state, BenchDevice bench_device) {
    std::size_t elements = static_cast<std::size_t>(state.range(0));
    auto device = make_device(bench_device);

    Tensor a({elements, 0, 0, 0}, device);
    Tensor b({elements, 0, 0, 0}, device);
    Tensor out({elements, 0, 0, 0}, device, 0.0f, true);

    TensorMath::add(a, b, out);

    for(auto _ : state) {
        TensorMath::add(a, b, out);
        benchmark::DoNotOptimize(out.data());
    }

    set_common_counters(state, bench_device, elements);
}

void matmul(benchmark::State& state, BenchDevice bench_device) {
    std::size_t m = static_cast<std::size_t>(state.range(0));
    std::size_t n = static_cast<std::size_t>(state.range(1));
    std::size_t k = static_cast<std::size_t>(state.range(2));
    auto device = make_device(bench_device);

    Tensor a({m, k, 0, 0}, device);
    Tensor b({k, n, 0, 0}, device);
    Tensor out({m, n, 0, 0}, device, 0.0f, true);

    TensorMath::matmul(a, b, out);

    for(auto _ : state) {
        TensorMath::matmul(a, b, out);
        benchmark::DoNotOptimize(out.data());
    }

    std::size_t output_elements = m * n;
    state.counters["M"] = static_cast<double>(m);
    state.counters["N"] = static_cast<double>(n);
    state.counters["K"] = static_cast<double>(k);
    state.counters["flop_per_second"] = benchmark::Counter(
        static_cast<double>(2 * m * n * k),
        benchmark::Counter::kIsIterationInvariantRate
    );
    set_common_counters(state, bench_device, output_elements);
}

void conv2d_forward(benchmark::State& state, BenchDevice bench_device) {
    std::size_t batch = static_cast<std::size_t>(state.range(0));
    std::size_t in_channels = static_cast<std::size_t>(state.range(1));
    std::size_t out_channels = static_cast<std::size_t>(state.range(2));
    std::size_t image_size = static_cast<std::size_t>(state.range(3));
    auto device = make_device(bench_device);

    constexpr std::size_t kernel_size = 3;
    constexpr std::size_t padding = 1;
    constexpr std::size_t stride = 1;

    Tensor input({batch, in_channels, image_size, image_size}, device);
    Conv2D conv(
        {out_channels, in_channels, kernel_size, kernel_size},
        device,
        {padding, padding},
        {stride, stride}
    );

    auto warmup = conv.forward(input);
    benchmark::DoNotOptimize(warmup->data());

    for(auto _ : state) {
        auto out = conv.forward(input);
        benchmark::DoNotOptimize(out->data());
    }

    std::size_t output_elements = batch * out_channels * image_size * image_size;
    std::size_t kernel_ops = 2 * in_channels * kernel_size * kernel_size;
    state.counters["batch"] = static_cast<double>(batch);
    state.counters["in_channels"] = static_cast<double>(in_channels);
    state.counters["out_channels"] = static_cast<double>(out_channels);
    state.counters["image_size"] = static_cast<double>(image_size);
    state.counters["flop_per_second"] = benchmark::Counter(
        static_cast<double>(output_elements * kernel_ops),
        benchmark::Counter::kIsIterationInvariantRate
    );
    set_common_counters(state, bench_device, output_elements);
}

void fully_connected_forward(benchmark::State& state, BenchDevice bench_device) {
    std::size_t batch = static_cast<std::size_t>(state.range(0));
    std::size_t input_dim = static_cast<std::size_t>(state.range(1));
    std::size_t output_dim = static_cast<std::size_t>(state.range(2));
    auto device = make_device(bench_device);

    Tensor input({batch, input_dim, 0, 0}, device);
    FullyConnected layer({input_dim, output_dim, 0, 0}, device);

    auto warmup = layer.forward(input);
    benchmark::DoNotOptimize(warmup->data());

    for(auto _ : state) {
        auto out = layer.forward(input);
        benchmark::DoNotOptimize(out->data());
    }

    std::size_t output_elements = batch * output_dim;
    state.counters["batch"] = static_cast<double>(batch);
    state.counters["input_dim"] = static_cast<double>(input_dim);
    state.counters["output_dim"] = static_cast<double>(output_dim);
    state.counters["flop_per_second"] = benchmark::Counter(
        static_cast<double>(2 * batch * input_dim * output_dim),
        benchmark::Counter::kIsIterationInvariantRate
    );
    set_common_counters(state, bench_device, output_elements);
}

void tensor_add_cpu(benchmark::State& state) {
    tensor_add(state, BenchDevice::CPU);
}

void matmul_cpu(benchmark::State& state) {
    matmul(state, BenchDevice::CPU);
}

void conv2d_forward_cpu(benchmark::State& state) {
    conv2d_forward(state, BenchDevice::CPU);
}

void fully_connected_forward_cpu(benchmark::State& state) {
    fully_connected_forward(state, BenchDevice::CPU);
}

#ifdef __USE_CUDA__
void tensor_add_gpu(benchmark::State& state) {
    tensor_add(state, BenchDevice::GPU);
}

void matmul_gpu(benchmark::State& state) {
    matmul(state, BenchDevice::GPU);
}

void conv2d_forward_gpu(benchmark::State& state) {
    conv2d_forward(state, BenchDevice::GPU);
}

void fully_connected_forward_gpu(benchmark::State& state) {
    fully_connected_forward(state, BenchDevice::GPU);
}
#endif

} // namespace

BENCHMARK(tensor_add_cpu)
    ->Name("tensor_add/CPU")
    ->Arg(1 << 12)
    ->Arg(1 << 16)
    ->Arg(1 << 20);

BENCHMARK(matmul_cpu)
    ->Name("matmul/CPU")
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256});

BENCHMARK(conv2d_forward_cpu)
    ->Name("conv2d_forward/CPU")
    ->Args({1, 1, 8, 28})
    ->Args({8, 1, 8, 28})
    ->Args({16, 8, 16, 28});

BENCHMARK(fully_connected_forward_cpu)
    ->Name("fully_connected_forward/CPU")
    ->Args({1, 784, 128})
    ->Args({32, 784, 128})
    ->Args({64, 1024, 256});

#ifdef __USE_CUDA__
BENCHMARK(tensor_add_gpu)
    ->Name("tensor_add/GPU")
    ->Arg(1 << 12)
    ->Arg(1 << 16)
    ->Arg(1 << 20);

BENCHMARK(matmul_gpu)
    ->Name("matmul/GPU")
    ->Args({64, 64, 64})
    ->Args({128, 128, 128})
    ->Args({256, 256, 256});

BENCHMARK(conv2d_forward_gpu)
    ->Name("conv2d_forward/GPU")
    ->Args({1, 1, 8, 28})
    ->Args({8, 1, 8, 28})
    ->Args({16, 8, 16, 28});

BENCHMARK(fully_connected_forward_gpu)
    ->Name("fully_connected_forward/GPU")
    ->Args({1, 784, 128})
    ->Args({32, 784, 128})
    ->Args({64, 1024, 256});
#endif
