
#include <core/ConvNeuralNetwork.hpp>
#include <stdexcept>
#include <cassert>

using ConvNN = ConvolutionalNeuralNetwork;

ConvNN::ConvolutionalNeuralNetwork(
    std::vector<Layer> layers
) : nn_layers(layers) {}


Tensor& ConvNN::forward(Tensor& input){
    assert(this->nn_layers.size() > 0);
    //Feed the input to the first layer
    this->nn_layers[0].forward(input);
    //And now, forward the output of the previous layer to the next layer
    for(std::size_t i = 1; i < this->nn_layers.size(); i++){
        this->nn_layers[i].forward(this->nn_layers[i-1]._out);
    }
    return this->nn_layers[this->nn_layers.size()-1]._out;
}

Tensor& ConvolutionalNeuralNetwork::backwards(Tensor& input){
    throw std::runtime_error("Not implemented");
}