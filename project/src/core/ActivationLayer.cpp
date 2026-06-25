#include <core/ActivationLayer.hpp>
#include <functional>
#include <stdexcept>
#include <math/TensorMath.hpp>

ActivationLayer::ActivationLayer(
    std::array<std::size_t, TENSOR_MAX_DIM> _input_shape,
    ActivationTypes _type,
    std::shared_ptr<Device> dev 
) : Layer(_input_shape,dev), type(_type) {}

Tensor& ActivationLayer::forward(Tensor& input){
    if(input.shape != this->_out.shape){
        throw std::runtime_error("Input does not match expected output size");
    }

    switch(this->type){
        case(ActivationTypes::ReLU):{
            TensorMath::reLU(input,this->_out);
        }

        case(ActivationTypes::Sigmoid) : {
            TensorMath::sigmoid(input,this->_out);
        }
    }

    return _out;
}

void ActivationLayer::update_weights(Tensor& incr){
    //This layer does not hold weights
    return;
}