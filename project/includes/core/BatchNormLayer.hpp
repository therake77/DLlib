#include <core/Layer.hpp>

class BatchNorm : public Layer{
private:
    //Parameters computed during training and fixed in inference mode
    Tensor _gamma;
    Tensor _beta;
    Tensor _means;  
    Tensor _variances;

    //Momentum (used during training)
    const float momentum = 0.1f;
    
public:
    ~BatchNorm() = default;

    Tensor& gamma = _gamma;
    Tensor& beta = _beta;
    Tensor& means = _means;
    Tensor& variances = _variances;

    BatchNorm(
        std::size_t channel_dim,
        std::shared_ptr<Device> dev
    );

    Tensor& forward(Tensor&) override;
    Tensor& forward(Tensor& , bool);
    void update_weights(Tensor&) override;
};