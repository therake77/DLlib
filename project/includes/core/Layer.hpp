#pragma once
#include <core/Tensor.hpp>

class Layer{
public:
    virtual ~Layer() = default;
    virtual Tensor forward(Tensor) = 0;
    virtual void update_weights(Tensor) = 0;
};


class Pooling : public Layer{
protected:
    int pool_size;
    int stride;
    virtual float pool_patch() = 0;
    virtual void backprop_patch() = 0;

public:
    Tensor forward(Tensor) override;
    void update_weights(Tensor) override;

};
