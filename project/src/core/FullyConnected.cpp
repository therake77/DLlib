#include <core/FullyConnected.hpp>
#include <stdexcept>
#include <math/TensorMath.hpp>

namespace {
    void _check_shape_dim(Tensor& t, std::size_t n_dim){
        if(t.shape.n_dim != n_dim){ throw std::runtime_error("Unexpected shape dimension"); }
        return;
    }
};

/*
    @brief Creates a dense layer.
    @param shape Array of shapes {`out`, `in`}
    @param dev Layer device
*/
FullyConnected::FullyConnected(
    std::array<size_t, TENSOR_MAX_DIM> shape,
    std::shared_ptr<Device> dev
) : _weights(shape,dev),_bias({shape[0]},dev),_grad(shape,dev),_out({shape[0]},dev) {
    _check_shape_dim(_weights,2);
    _check_shape_dim(_grad,2);
    _check_shape_dim(_bias,1);
    _check_shape_dim(_out,1);
}

Tensor& FullyConnected::forward(Tensor& input){    
    _check_shape_dim(input,1);
    if(input.shape[0] != this->weigths.shape[1]){ throw std::runtime_error("Unexpected input shape dimension"); }
    
    TensorMath::matmul(
        this->weigths,
        input,
        this->_out
    );

    TensorMath::add(
        this->_out,
        this->_bias,
        this->_out
    );

    return _out;
}

void FullyConnected::update_weights(Tensor& input){
    _check_shape_dim(input,2);
    TensorMath::add(
        this->weigths,
        input,
        this->weigths
    );
}

void FullyConnected::update_bias(Tensor& input){
    _check_shape_dim(input,1);
    TensorMath::add(
        this->_bias,
        input,
        this->_bias
    );
}