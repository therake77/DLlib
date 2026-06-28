#include <core/Layer.hpp>
#include <list>

class ConvolutionalNeuralNetwork{
private:
    std::vector<std::unique_ptr<Layer>> nn_layers;

public:
    ConvolutionalNeuralNetwork(
        std::vector<std::unique_ptr<Layer>>&& layers
    );

    std::shared_ptr<Tensor> forward( const Tensor& );
    std::shared_ptr<Tensor> backward( const Tensor& );

};