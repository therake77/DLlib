#include <core/Layer.hpp>

class FullyConnected : public Layer{
private:
    Tensor _weights;
    Tensor _bias;
    Tensor _grad;

public:
    Tensor& weigths = _weights;
    Tensor& grad = _grad;

    FullyConnected(
        std::array<size_t, TENSOR_MAX_DIM> weight_shape,
        std::shared_ptr<Device> dev
    );

    Tensor& forward(Tensor&) override;
    void update_weights(Tensor&) override;

    void update_bias(Tensor&);
};
