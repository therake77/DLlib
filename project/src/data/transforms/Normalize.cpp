#include <data/transforms/Normalize.hpp>

#include <stdexcept>
#include <utility>

Normalize::Normalize(float mean, float std)
    : Normalize(std::vector<float>{mean}, std::vector<float>{std}) {
}

Normalize::Normalize(std::vector<float> mean, std::vector<float> std)
    : _mean(std::move(mean)),
      _std(std::move(std)) {
    if(_mean.empty() || _std.empty()) {
        throw std::runtime_error("Error: Normalize mean and std cannot be empty");
    }

    if(_mean.size() != 1 && _mean.size() != _std.size()) {
        throw std::runtime_error("Error: Normalize mean and std sizes must match");
    }

    for(float value : _std) {
        if(value == 0.0f) {
            throw std::runtime_error("Error: Normalize std values must be non-zero");
        }
    }
}

Tensor Normalize::operator()(const Tensor& input) const {
    return apply(input);
}

Tensor Normalize::apply(const Tensor& input) const {
    if(input.shape.n_dim != 4) {
        throw std::runtime_error("Error: Normalize expects a 4D tensor with shape (N, C, H, W)");
    }

    std::size_t batch = input.shape[0];
    std::size_t channels = input.shape[1];
    std::size_t height = input.shape[2];
    std::size_t width = input.shape[3];

    if(_mean.size() != 1 && _mean.size() != channels) {
        throw std::runtime_error("Error: Normalize mean size must be 1 or match tensor channels");
    }

    if(_std.size() != 1 && _std.size() != channels) {
        throw std::runtime_error("Error: Normalize std size must be 1 or match tensor channels");
    }

    std::shared_ptr<Device> original_device = input.getDevice();
    Tensor input_cpu(input, DeviceManager::get_cpu_device());
    Tensor output_cpu(input_cpu, DeviceManager::get_cpu_device());

    const float* input_data = input_cpu.data();
    float* output_data = output_cpu.data();

    std::size_t spatial = height * width;
    std::size_t sample_size = channels * spatial;

    for(std::size_t n = 0; n < batch; ++n) {
        for(std::size_t c = 0; c < channels; ++c) {
            float mean = _channel_value(_mean, c);
            float std = _channel_value(_std, c);

            for(std::size_t i = 0; i < spatial; ++i) {
                std::size_t idx = n * sample_size + c * spatial + i;
                output_data[idx] = (input_data[idx] - mean) / std;
            }
        }
    }

    if(original_device->get_type() != DeviceType::CPU) {
        output_cpu.to(original_device);
    }

    return output_cpu;
}

float Normalize::_channel_value(
    const std::vector<float>& values,
    std::size_t channel
) {
    if(values.size() == 1) {
        return values[0];
    }

    return values[channel];
}
