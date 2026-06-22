#include <core/Layer.hpp>

class Conv2D : public Layer{
private:
    std::shared_ptr<Device> device;
    Tensor _kernels;
    Tensor _grad;
    std::array<size_t,2> padding;
    std::array<size_t,2> strides;
public:
    Tensor& kernels = _kernels;
    Tensor& grad = _grad;
    
    ~Conv2D() override = default;

    Conv2D(std::array<size_t, 4> shape ,std::shared_ptr<Device> dev, std::array<size_t,2> _padding, std::array<size_t,2> _strides);
    Tensor forward(Tensor&) override;
    void update_weights(Tensor&) override;
};

class FullyConnected : public Layer{
private:
    Tensor _weights;
    Tensor _bias;
    Tensor _grad;
public:
    Tensor& weigths = _weights;
    Tensor& grad = _grad;
    Tensor forward(Tensor&) override;
    void update_weights(Tensor&) override;

};
