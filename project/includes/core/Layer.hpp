#pragma once
#include <core/Tensor.hpp>

class Layer{
public:
    virtual ~Layer() = default;
    virtual Tensor forward(Tensor&) = 0;
    virtual void update_weights(Tensor&) = 0;
};


class Pooling2D : public Layer{
protected:
    std::array<std::size_t, 2> window_shape;
    std::array<std::size_t, 2> stride;
    std::array<std::size_t, 2> padding;
    virtual float pool_patch() = 0;
    virtual void backprop_patch() = 0;

public:
    Tensor forward(Tensor&) override;
    void update_weights(Tensor&) override;

};
