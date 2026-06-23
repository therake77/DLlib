#pragma once
#include <core/Tensor.hpp>

class Layer{
public:
    virtual ~Layer() = default;
    virtual Tensor forward(Tensor&) = 0;
    virtual void update_weights(Tensor&) = 0;
};