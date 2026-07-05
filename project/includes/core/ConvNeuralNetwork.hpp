#include <core/Layer.hpp>
#include <core/OutputLayer.hpp>
#include <list>

class ConvolutionalNeuralNetwork{
private:
    std::vector<std::unique_ptr<Layer>> nn_layers;

public:
    ConvolutionalNeuralNetwork(
        std::vector<std::unique_ptr<Layer>>&& layers
    );

    std::shared_ptr<Tensor> forward( const Tensor& in );
    std::shared_ptr<Tensor> forward( const Tensor& in, const Tensor& labels);
    
    std::shared_ptr<Tensor> backward( const Tensor& original_in ,const Tensor& labels );

    void set_random_weights();
    void set_mode(LayerMode);
    void update_weights(float);
    void training(Tensor&, Tensor&);
};