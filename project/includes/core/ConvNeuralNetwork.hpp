#include <core/Layer.hpp>
#include <list>

class ConvolutionalNeuralNetwork{
private:
    std::vector<Layer> nn_layers;

public:
    ConvolutionalNeuralNetwork(
        std::vector<Layer> layers
    );

    std::shared_ptr<Tensor> forward( const Tensor& );
    std::shared_ptr<Tensor> backward( const Tensor& );

};