#include <data/transforms/Resize.hpp>

#include <external/stb_image_resize2.h>

#include <limits>
#include <stdexcept>
#include <vector>

Resize::Resize(std::size_t height, std::size_t width)
    : _height(height),
      _width(width) {
    if(_height == 0 || _width == 0) {
        throw std::runtime_error("Error: Resize dimensions must be greater than zero");
    }
}

Tensor Resize::operator()(const Tensor& input) const {
    return apply(input);
}

Tensor Resize::apply(const Tensor& input) const {
    if(input.shape.n_dim != 4) {
        throw std::runtime_error("Error: Resize expects a 4D tensor with shape (N, C, H, W)");
    }

    std::size_t batch = input.shape[0];
    std::size_t channels = input.shape[1];
    std::size_t input_height = input.shape[2];
    std::size_t input_width = input.shape[3];

    if(channels != 1 && channels != 3 && channels != 4) {
        throw std::runtime_error("Error: Resize supports tensors with 1, 3, or 4 channels");
    }

    if(input_height == 0 || input_width == 0) {
        throw std::runtime_error("Error: Resize input dimensions must be greater than zero");
    }

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

    std::size_t input_sample_size = channels * input_height * input_width;
    std::size_t output_sample_size = channels * _height * _width;

    for(std::size_t n = 0; n < batch; ++n) {
        _resize_single_image(
            input_data + n * input_sample_size,
            output_data + n * output_sample_size,
            channels,
            input_height,
            input_width,
            _height,
            _width
        );
    }

    if(original_device->get_type() != DeviceType::CPU) {
        output_cpu.to(original_device);
    }

    return output_cpu;
}

void Resize::_resize_single_image(
    const float* input,
    float* output,
    std::size_t channels,
    std::size_t input_height,
    std::size_t input_width,
    std::size_t output_height,
    std::size_t output_width
) {
    if(input_height > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       input_width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       output_height > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
       output_width > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Error: Resize dimensions exceed stb_image_resize limits");
    }

    std::size_t input_spatial = input_height * input_width;
    std::size_t output_spatial = output_height * output_width;

    std::vector<float> input_hwc(input_height * input_width * channels);
    std::vector<float> output_hwc(output_height * output_width * channels);

    for(std::size_t y = 0; y < input_height; ++y) {
        for(std::size_t x = 0; x < input_width; ++x) {
            for(std::size_t c = 0; c < channels; ++c) {
                std::size_t chw_idx = c * input_spatial + y * input_width + x;
                std::size_t hwc_idx = (y * input_width + x) * channels + c;
                input_hwc[hwc_idx] = input[chw_idx];
            }
        }
    }

    stbir_pixel_layout pixel_layout = static_cast<stbir_pixel_layout>(channels);
    float* resized = stbir_resize_float_linear(
        input_hwc.data(),
        static_cast<int>(input_width),
        static_cast<int>(input_height),
        0,
        output_hwc.data(),
        static_cast<int>(output_width),
        static_cast<int>(output_height),
        0,
        pixel_layout
    );

    if(resized == nullptr) {
        throw std::runtime_error("Error: Failed to resize image");
    }

    for(std::size_t y = 0; y < output_height; ++y) {
        for(std::size_t x = 0; x < output_width; ++x) {
            for(std::size_t c = 0; c < channels; ++c) {
                std::size_t hwc_idx = (y * output_width + x) * channels + c;
                std::size_t chw_idx = c * output_spatial + y * output_width + x;
                output[chw_idx] = output_hwc[hwc_idx];
            }
        }
    }
}
