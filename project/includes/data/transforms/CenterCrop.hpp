#pragma once

#include <data/transforms/Transform.hpp>

#include <cstddef>

class CenterCrop : public Transform {
public:
    CenterCrop(std::size_t height, std::size_t width);

    Tensor apply(const Tensor& input) const override;
    Tensor operator()(const Tensor& input) const;

    std::size_t height() const { return _height; }
    std::size_t width() const { return _width; }

private:
    std::size_t _height;
    std::size_t _width;
};
