#include <core/FullyConnected.hpp>
#include <stdexcept>
#include <math/TensorMath.hpp>
#include <cassert>

/*
    @brief Creates a dense layer.
    @param shape Array of shapes {`in`,`out`}
    @param dev Layer device
*/
FullyConnected::FullyConnected(
    std::array<size_t, TENSOR_MAX_DIM> shape,
    std::shared_ptr<Device> dev
) : _weights(shape,dev),_bias({1,shape[0]},dev), Layer(dev) {
    if(Shape(shape).n_dim != 2){ throw std::runtime_error("Error: Only valid 2-dimensional shapes"); }
    this->input_dim = shape[0];
    this->output_dim = shape[1];
    this->_bias_grad = std::make_unique<Tensor>(Tensor({1,this->output_dim},dev,0.0f,true));
}

std::shared_ptr<Tensor> FullyConnected::forward(const Tensor& in){
    //IMPORTANT: in is 2-dimensional (N,F) (batch, flatten)    
    //Well, everyone knows the forward pass Y= XW + b
    __check_initialization(this->_forward_out,Shape({in.shape[0],this->output_dim}));
    TensorMath::matmul(
        in,
        this->_weights,
        *(this->_forward_out)
    );
    TensorMath::add(
        this->_bias,
        *(this->_forward_out),
        *(this->_forward_out)
    );
    return this->_forward_out;
}

std::shared_ptr<Tensor> FullyConnected::backward(const Tensor& backward_grad, const Tensor& forward_input){
    assert(backward_grad.shape.n_dim == 2);
    __check_initialization(this->_bias_grad,Shape({this->output_dim}));
    __check_initialization(this->_grad,this->_weights.shape);
    //Transpose _forward_out
    this->_forward_out->shape.transpose();
    //Compute weight gradient
    TensorMath::matmul(
        *(this->_forward_out),
        backward_grad,
        *(this->_grad)
    );
    //Compute (TODO)
}


void FullyConnected::update_weight(){
    assert(this->_bias_grad.get() != nullptr);
    assert(this->_grad.get() != nullptr);
    TensorMath::add(
        this->_weights,
        *(this->_grad),
        this->_weights
    );
    TensorMath::add(
        this->_bias,
        *(this->_bias_grad),
        this->_bias
    );
    return;
}
