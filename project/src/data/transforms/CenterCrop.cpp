#include <data/transforms/CenterCrop.hpp>

#include <stdexcept>

CenterCrop::CenterCrop(std::size_t height, std::size_t width)
    : _height(height),
      _width(width) {
    if(_height == 0 || _width == 0) {
        throw std::runtime_error("Error: CenterCrop dimensions must be greater than zero");
    }
}

Tensor CenterCrop::operator()(const Tensor& input) const {
    return apply(input);
}

Tensor CenterCrop::apply(const Tensor& input) const {
    if(input.shape.n_dim != 4) {
        throw std::runtime_error("Error: CenterCrop expects a 4D tensor with shape (N, C, H, W)");
    }

    std::size_t batch = input.shape[0];
    std::size_t channels = input.shape[1];
    std::size_t input_height = input.shape[2];
    std::size_t input_width = input.shape[3];

    if(_height > input_height || _width > input_width) {
        throw std::runtime_error("Error: CenterCrop dimensions cannot exceed input dimensions");
    }

    std::size_t offset_y = (input_height - _height) / 2;
    std::size_t offset_x = (input_width - _width) / 2;

    std::shared_ptr<Device> original_device = input.getDevice();
    Tensor input_cpu(input, DeviceManager::get_cpu_device());
    Tensor output_cpu(
        {batch, channels, _height, _width},
        DeviceManager::get_cpu_device(),
        0.0f,
        true
    );

    const float* input_data = input_cpu.data();
    float* output_data = output_cpu.data();

    std::size_t input_spatial = input_height * input_width;
    std::size_t output_spatial = _height * _width;
    std::size_t input_sample_size = channels * input_spatial;
    std::size_t output_sample_size = channels * output_spatial;

    for(std::size_t n = 0; n < batch; ++n) {
        for(std::size_t c = 0; c < channels; ++c) {
            for(std::size_t y = 0; y < _height; ++y) {
                for(std::size_t x = 0; x < _width; ++x) {
                    std::size_t input_idx =
                        n * input_sample_size +
                        c * input_spatial +
                        (offset_y + y) * input_width +
                        (offset_x + x);

                    std::size_t output_idx =
                        n * output_sample_size +
                        c * output_spatial +
                        y * _width +
                        x;

                    output_data[output_idx] = input_data[input_idx];
                }
            }
        }
    }

    if(original_device->get_type() != DeviceType::CPU) {
        output_cpu.to(original_device);
    }

    return output_cpu;
}
