#include <core/Layer.hpp>

class BatchNorm : public Layer{
private:
    //Parameters computed during training and fixed in inference mode
    Tensor _gamma;  //Its gradient is weights
    Tensor _beta;   
    Tensor _beta_grad;  //This is the gradient for _beta
    //Placeholders for training forward pass
    Tensor _means;  
    Tensor _variances;
    Tensor _sqrt_variances;
    //Actual means and variances used in inference
    Tensor _running_means;
    Tensor _running_variances;

    //Placeholder for normalized input
    std::unique_ptr<Tensor> _normalized_input;

    //Momentum (used during training)
    const float momentum = 0.1f;
    
public:
    ~BatchNorm() = default;

    Tensor& gamma = _gamma;
    Tensor& beta = _beta;
    Tensor& running_means =  _running_means;
    Tensor& running_variances =  _running_variances;


    BatchNorm(
        std::size_t channel_dim,
        std::shared_ptr<Device> dev
    );

    std::shared_ptr<Tensor> forward( const Tensor& ) override;
    std::shared_ptr<Tensor> backward( const Tensor&, const Tensor&  ) override;
    void update_weight() override;
};