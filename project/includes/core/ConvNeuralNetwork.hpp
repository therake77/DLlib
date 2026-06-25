#include <core/Layer.hpp>
#include <list>

class ConvolutionalNeuralNetwork{
private:
    std::vector<Layer> nn_layers;

public:
    ConvolutionalNeuralNetwork(
        std::vector<Layer> layers
    );

    Tensor& forward(Tensor& input);
    Tensor& backwards(Tensor& input);

};