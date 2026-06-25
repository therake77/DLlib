#pragma once
#include <core/Tensor.hpp>

class Layer{
public:
    Tensor _out;
    
    Layer(
        std::array<std::size_t,TENSOR_MAX_DIM> shape,
        std::shared_ptr<Device> dev
    ) : _out(shape,dev) {}
    
    virtual ~Layer() = default;
    virtual Tensor& forward(Tensor&) = 0;
    virtual void update_weights(Tensor&) = 0;
};