#include <core/Pooling.hpp>
#include <math/TensorMath.hpp>
#include <stdexcept>
#include <cassert>

MaxPooling::MaxPooling(
    std::array<std::size_t, 2> _window_shape,
    std::array<std::size_t, 2> _stride,
    std::array<std::size_t, 2> _padding,
    std::shared_ptr<Device> dev
) : Layer(dev), window_shape(_window_shape), stride(_stride), padding(_padding){}


std::shared_ptr<Tensor> MaxPooling::forward(const Tensor& in){
    //Dimension shapes
    std::size_t batch_dim = in.shape[0];
    std::size_t channel_dim = in.shape[1];
    std::size_t out_h = (in.shape[2] - this->window_shape[0] + 2 * padding[0])/stride[0] + 1;
    std::size_t out_w = (in.shape[3] - this->window_shape[1] + 2 * padding[1])/stride[1] + 1;
    //Check initializations
    __check_initialization(_mask,in.shape);
    __check_initialization(this->_forward_out, Shape({batch_dim,channel_dim,out_h,out_w}));
    //Call max_pooling
    TensorMath::max_pooling2d(
        in,
        *(this->_forward_out),
        *(this->_mask),
        this->stride,
        this->window_shape,
        this->padding
    );
    //End
    return this->_forward_out;
}

std::shared_ptr<Tensor> MaxPooling::backward( const Tensor& backward_grad, const Tensor& forward_input ){
    assert(this->_mask.get() != nullptr);
    __check_initialization(this->_backward_out, forward_input.shape );
    TensorMath::inv_max_pooling2D(
        backward_grad,
        this->stride,
        this->window_shape,
        *(this->_mask),
        *(this->_backward_out),
        this->padding
    );
    return this->_backward_out;
}


void MaxPooling::update_weight(float){
    //Do nothing
    return;
}

AvgPooling::AvgPooling(
    std::array<std::size_t, 2> _window_shape,
    std::array<std::size_t, 2> _stride,
    std::array<std::size_t, 2> _padding,
    std::shared_ptr<Device> dev
) : Layer(dev), window_shape(_window_shape), stride(_stride), padding(_padding) {}


std::shared_ptr<Tensor> AvgPooling::forward(const Tensor& in){
    
    //Dimension shapes
    std::size_t batch_dim = in.shape[0];
    std::size_t channel_dim = in.shape[1];
    std::size_t out_h = (in.shape[2] + 2 * padding[0] - this->window_shape[0])/stride[0] + 1;
    std::size_t out_w = (in.shape[3] + 2 * padding[1] - this->window_shape[1])/stride[1] + 1;
    //Check initializations
    __check_initialization(this->_forward_out, Shape({batch_dim,channel_dim,out_h,out_w}));
    //Call max_pooling
    TensorMath::avg_pooling2d(
        in,
        *(this->_forward_out),
        this->stride,
        this->window_shape,
        this->padding
    );
    //End
    return this->_forward_out;
}

std::shared_ptr<Tensor> AvgPooling::backward( const Tensor& backward_grad, const Tensor& forward_input ){
    __check_initialization(this->_backward_out, forward_input.shape );
    TensorMath::inv_avg_pooling2D(
        backward_grad,
        this->stride,
        this->window_shape,
        *(this->_backward_out),
        this->padding
    );
    return this->_backward_out;
}

void AvgPooling::update_weight(float){
    //No weights to update
    return;
}
