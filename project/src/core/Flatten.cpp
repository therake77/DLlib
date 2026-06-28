#include <core/Flatten.hpp>

Flatten::Flatten(
    std::shared_ptr<Device> dev
) : Layer(dev){}

std::shared_ptr<Tensor> Flatten::forward( const Tensor& in ){
    //Copy in to the forward_out buffer
    using dim_t = std::size_t;
    dim_t N = in.shape[0];
    dim_t C = in.shape[1];
    dim_t H = in.shape[2];
    dim_t W = in.shape[3];

    __check_initialization(this->_forward_out,Shape({N,C*H*W}));
    this->_forward_out->copy_from(in);

    return this->_forward_out;
}

std::shared_ptr<Tensor> Flatten::backward( const Tensor& backward_grad, const Tensor& forward_in ){
    __check_initialization(this->_backward_out,forward_in.shape);
    this->_backward_out->copy_from(backward_grad);
    return this->_backward_out;
}

void Flatten::update_weight(){
    return;
}
