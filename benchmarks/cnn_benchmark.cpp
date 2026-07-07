#include <benchmark/benchmark.h>

#include <core/ConvLayer.hpp>
#include <core/ActivationLayer.hpp>
#include <core/ConvNeuralNetwork.hpp>
#include <core/FullyConnected.hpp>
#include <core/Flatten.hpp>
#include <core/OutputLayer.hpp>
#include <core/Pooling.hpp>
#include <core/Tensor.hpp>
#include <hardware/DeviceManager.hpp>
#include <math/TensorMath.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

Tensor make_one_hot_labels(
    std::size_t batch,
    std::size_t classes,
    std::size_t batch_index,
    const std::shared_ptr<Device>& device
) {
    std::vector<float> labels(batch * classes, 0.0f);
    for(std::size_t n = 0; n < batch; ++n) {
        labels[n * classes + ((n + batch_index) % classes)] = 1.0f;
    }

    return Tensor({batch, classes, 0, 0}, labels.data(), device);
}

struct SyntheticBatch {
    Tensor input;
    Tensor labels;
};

std::vector<SyntheticBatch> make_synthetic_batches(
    std::size_t batch,
    std::size_t batch_count,
    const std::shared_ptr<Device>& device
) {
    std::vector<SyntheticBatch> batches;
    batches.reserve(batch_count);

    for(std::size_t index = 0; index < batch_count; ++index) {
        batches.push_back(SyntheticBatch{
            Tensor({batch, 1, 28, 28}, device),
            make_one_hot_labels(batch, 10, index, device)
        });
    }

    return batches;
}

ConvolutionalNeuralNetwork make_mnist_cnn(const std::shared_ptr<Device>& device) {
    std::vector<std::unique_ptr<Layer>> layers;

    layers.push_back(std::make_unique<Conv2D>(
        std::array<std::size_t, 4>{32, 1, 3, 3},
        device,
        std::array<std::size_t, 2>{0, 0},
        std::array<std::size_t, 2>{1, 1}
    ));
    layers.push_back(std::make_unique<ActivationLayer>(ActivationTypes::LeakyReLU, device));
    layers.push_back(std::make_unique<MaxPooling>(
        std::array<std::size_t, 2>{2, 2},
        std::array<std::size_t, 2>{2, 2},
        std::array<std::size_t, 2>{0, 0},
        device
    ));

    layers.push_back(std::make_unique<Conv2D>(
        std::array<std::size_t, 4>{64, 32, 3, 3},
        device,
        std::array<std::size_t, 2>{0, 0},
        std::array<std::size_t, 2>{1, 1}
    ));
    layers.push_back(std::make_unique<ActivationLayer>(ActivationTypes::LeakyReLU, device));
    layers.push_back(std::make_unique<MaxPooling>(
        std::array<std::size_t, 2>{2, 2},
        std::array<std::size_t, 2>{2, 2},
        std::array<std::size_t, 2>{0, 0},
        device
    ));

    layers.push_back(std::make_unique<Flatten>(device));
    layers.push_back(std::make_unique<FullyConnected>(
        std::array<std::size_t, TENSOR_MAX_DIM>{1600, 128, 0, 0},
        device
    ));
    layers.push_back(std::make_unique<ActivationLayer>(ActivationTypes::LeakyReLU, device));
    layers.push_back(std::make_unique<FullyConnected>(
        std::array<std::size_t, TENSOR_MAX_DIM>{128, 10, 0, 0},
        device
    ));
    layers.push_back(std::make_unique<OutputLayer>(device, 10));

    ConvolutionalNeuralNetwork cnn(std::move(layers));
    cnn.set_random_weights();
    return cnn;
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

void tensor_transfer(
    benchmark::State& state,
    BenchDevice source_device,
    BenchDevice target_device
) {
    std::size_t elements = static_cast<std::size_t>(state.range(0));
    auto source = make_device(source_device);
    auto target = make_device(target_device);

    for(auto _ : state) {
        state.PauseTiming();
        auto tensor = std::make_unique<Tensor>(
            std::array<std::size_t, TENSOR_MAX_DIM>{elements, 0, 0, 0},
            source
        );
        state.ResumeTiming();

        tensor->to(target);
        benchmark::DoNotOptimize(tensor->data());

        state.PauseTiming();
        tensor.reset();
        state.ResumeTiming();
    }

    state.counters["bytes"] = static_cast<double>(elements * sizeof(float));
    state.counters["bytes_per_second"] = benchmark::Counter(
        static_cast<double>(elements * sizeof(float)),
        benchmark::Counter::kIsIterationInvariantRate
    );
    state.counters["elements"] = static_cast<double>(elements);
    std::string label = std::string(device_name(source_device)) + "_to_" + device_name(target_device);
    state.SetLabel(label);
}

void cnn_forward(benchmark::State& state, BenchDevice bench_device) {
    std::size_t batch = static_cast<std::size_t>(state.range(0));
    auto device = make_device(bench_device);

    Tensor input({batch, 1, 28, 28}, device);
    auto cnn = make_mnist_cnn(device);
    cnn.set_mode(LayerMode::Inference);

    auto warmup = cnn.forward(input);
    benchmark::DoNotOptimize(warmup->data());

    for(auto _ : state) {
        auto out = cnn.forward(input);
        benchmark::DoNotOptimize(out->data());
    }

    state.counters["batch"] = static_cast<double>(batch);
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<double>(batch),
        benchmark::Counter::kIsIterationInvariantRate
    );
    set_common_counters(state, bench_device, batch * 10);
}

void cnn_training_step(benchmark::State& state, BenchDevice bench_device) {
    std::size_t batch = static_cast<std::size_t>(state.range(0));
    auto device = make_device(bench_device);

    constexpr std::size_t synthetic_batch_count = 16;
    auto batches = make_synthetic_batches(batch, synthetic_batch_count, device);
    auto cnn = make_mnist_cnn(device);
    cnn.set_mode(LayerMode::Training);

    constexpr float learning_rate = -0.01f;

    auto warmup = cnn.forward(batches[0].input, batches[0].labels);
    cnn.backward(batches[0].input, batches[0].labels);
    cnn.update_weights(learning_rate);
    benchmark::DoNotOptimize(warmup->data());

    std::size_t batch_index = 0;
    for(auto _ : state) {
        const auto& current = batches[batch_index];
        auto loss = cnn.forward(current.input, current.labels);
        cnn.backward(current.input, current.labels);
        cnn.update_weights(learning_rate);
        benchmark::DoNotOptimize(loss->data());
        batch_index = (batch_index + 1) % batches.size();
    }

    state.counters["batch"] = static_cast<double>(batch);
    state.counters["synthetic_batches"] = static_cast<double>(synthetic_batch_count);
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<double>(batch),
        benchmark::Counter::kIsIterationInvariantRate
    );
    set_common_counters(state, bench_device, batch);
}

void cnn_training_loop(benchmark::State& state, BenchDevice bench_device) {
    std::size_t batch = static_cast<std::size_t>(state.range(0));
    std::size_t training_steps = static_cast<std::size_t>(state.range(1));
    auto device = make_device(bench_device);

    constexpr std::size_t synthetic_batch_count = 16;
    auto batches = make_synthetic_batches(batch, synthetic_batch_count, device);
    auto cnn = make_mnist_cnn(device);
    cnn.set_mode(LayerMode::Training);

    constexpr float learning_rate = -0.01f;

    auto warmup = cnn.forward(batches[0].input, batches[0].labels);
    cnn.backward(batches[0].input, batches[0].labels);
    cnn.update_weights(learning_rate);
    benchmark::DoNotOptimize(warmup->data());

    std::size_t batch_index = 0;
    for(auto _ : state) {
        std::shared_ptr<Tensor> loss;
        for(std::size_t step = 0; step < training_steps; ++step) {
            const auto& current = batches[batch_index];
            loss = cnn.forward(current.input, current.labels);
            cnn.backward(current.input, current.labels);
            cnn.update_weights(learning_rate);
            batch_index = (batch_index + 1) % batches.size();
        }
        benchmark::DoNotOptimize(loss->data());
    }

    state.counters["batch"] = static_cast<double>(batch);
    state.counters["training_steps"] = static_cast<double>(training_steps);
    state.counters["synthetic_batches"] = static_cast<double>(synthetic_batch_count);
    state.counters["samples_per_second"] = benchmark::Counter(
        static_cast<double>(batch * training_steps),
        benchmark::Counter::kIsIterationInvariantRate
    );
    set_common_counters(state, bench_device, batch * training_steps);
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

void cnn_forward_cpu(benchmark::State& state) {
    cnn_forward(state, BenchDevice::CPU);
}

void cnn_training_step_cpu(benchmark::State& state) {
    cnn_training_step(state, BenchDevice::CPU);
}

void cnn_training_loop_cpu(benchmark::State& state) {
    cnn_training_loop(state, BenchDevice::CPU);
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

void cnn_forward_gpu(benchmark::State& state) {
    cnn_forward(state, BenchDevice::GPU);
}

void cnn_training_step_gpu(benchmark::State& state) {
    cnn_training_step(state, BenchDevice::GPU);
}

void cnn_training_loop_gpu(benchmark::State& state) {
    cnn_training_loop(state, BenchDevice::GPU);
}

void tensor_transfer_cpu_to_gpu(benchmark::State& state) {
    tensor_transfer(state, BenchDevice::CPU, BenchDevice::GPU);
}

void tensor_transfer_gpu_to_cpu(benchmark::State& state) {
    tensor_transfer(state, BenchDevice::GPU, BenchDevice::CPU);
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

BENCHMARK(cnn_forward_cpu)
    ->Name("cnn_forward/CPU")
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(cnn_training_step_cpu)
    ->Name("cnn_training_step/CPU")
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(cnn_training_loop_cpu)
    ->Name("cnn_training_loop/CPU")
    ->Args({64, 10});

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

BENCHMARK(cnn_forward_gpu)
    ->Name("cnn_forward/GPU")
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(cnn_training_step_gpu)
    ->Name("cnn_training_step/GPU")
    ->Arg(1)
    ->Arg(8)
    ->Arg(16)
    ->Arg(32)
    ->Arg(64);

BENCHMARK(cnn_training_loop_gpu)
    ->Name("cnn_training_loop/GPU")
    ->Args({64, 10});

BENCHMARK(tensor_transfer_cpu_to_gpu)
    ->Name("tensor_transfer/CPU_to_GPU")
    ->Arg(1 << 12)
    ->Arg(1 << 16)
    ->Arg(1 << 20);

BENCHMARK(tensor_transfer_gpu_to_cpu)
    ->Name("tensor_transfer/GPU_to_CPU")
    ->Arg(1 << 12)
    ->Arg(1 << 16)
    ->Arg(1 << 20);
#endif
