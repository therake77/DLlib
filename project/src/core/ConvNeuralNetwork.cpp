#include <core/ConvNeuralNetwork.hpp>
#include <stdexcept>
#include <cassert>

using ConvNN = ConvolutionalNeuralNetwork;

ConvNN::ConvolutionalNeuralNetwork(
    std::vector<std::unique_ptr<Layer>>&& layers
) : nn_layers(std::move(layers)) {}


std::shared_ptr<Tensor> ConvNN::forward(const Tensor& input){
    assert(this->nn_layers.size() > 0);
    //Feed the input to the first layer
    this->nn_layers[0]->forward(input);
    //And now, forward the output of the previous layer to the next layer
    for(std::size_t i = 1; i < this->nn_layers.size(); i++){
        this->nn_layers[i]->forward(*(this->nn_layers[i-1]->_forward_out));
    }
    return this->nn_layers[this->nn_layers.size()-1]->_forward_out;
}

std::shared_ptr<Tensor> ConvNN::forward( const Tensor& input , const Tensor& labels){
    assert(this->nn_layers.size() > 2);
    std::size_t n_layers = this->nn_layers.size();
    //Feed the input to the first layer
    this->nn_layers[0]->forward(input);
    //And now, forward the output of the previous layer to the next layer
    for(std::size_t i = 1; i < this->nn_layers.size()-1; i++){
        this->nn_layers[i]->forward(*(this->nn_layers[i-1]->_forward_out));
    }
    //And for the output layer:
    auto last_layer = dynamic_cast<OutputLayer&>(*(this->nn_layers[n_layers-1]));
    last_layer.forward(*(this->nn_layers[n_layers-2]->_forward_out),labels);
    return this->nn_layers[this->nn_layers.size()-1]->_forward_out;
}

std::shared_ptr<Tensor> ConvolutionalNeuralNetwork::backward(const Tensor& input, const Tensor& labels){
    assert(this->nn_layers.size() > 2);
    std::size_t n_layers = this->nn_layers.size();
    auto last_layer = dynamic_cast<OutputLayer&>(*(this->nn_layers[n_layers-1]));
    last_layer.backward(labels,*(this->nn_layers[n_layers-2]->_forward_out));
    for(std::size_t i = n_layers-1; i--> 1; ){
        this->nn_layers[i]->backward(*(this->nn_layers[i+1]->_backward_out),*(this->nn_layers[i-1]->_forward_out));
    }
    
    this->nn_layers[0]->backward(*(this->nn_layers[1]->_backward_out),input);
    return this->nn_layers[0]->_backward_out;
}

void ConvNN::update_weights( float learning_rate ){
    for(std::size_t i = 0; i < this->nn_layers.size(); i++){
        this->nn_layers[i]->update_weight(learning_rate);
    }
    return;
}

void ConvNN::set_mode(LayerMode mode){
    for(std::size_t i = 0; i < this->nn_layers.size() ; i++){
        nn_layers[i]->set_mode(mode);
    }
    return;
}

void ConvNN::training(Tensor& batched_dataset, Tensor& label ){
    
    return;
}