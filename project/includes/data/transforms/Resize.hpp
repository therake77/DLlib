#pragma once

#include <core/Tensor.hpp>
#include <data/transforms/Transform.hpp>

#include <cstddef>

class Resize : public Transform {
public:
    Resize(std::size_t height, std::size_t width);

    Tensor apply(const Tensor& input) const override;
    Tensor operator()(const Tensor& input) const;

    std::size_t height() const { return _height; }
    std::size_t width() const { return _width; }

private:
    std::size_t _height;
    std::size_t _width;

    static void _resize_single_image(
        const float* input,
        float* output,
        std::size_t channels,
        std::size_t input_height,
        std::size_t input_width,
        std::size_t output_height,
        std::size_t output_width
    );
};
