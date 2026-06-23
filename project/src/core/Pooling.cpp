#include <core/Pooling.hpp>
#include <stdexcept>

Tensor MaxPooling::forward(Tensor& input){
    
    auto device = input.getDevice();

    std::size_t batch_dim = input.shape[0];
    std::size_t channel_dim = input.shape[1];
    std::size_t out_h = (input.shape[2] - this->window_shape[0] + 2 * padding[0])/stride[0] + 1;
    std::size_t out_w = (input.shape[3] - this->window_shape[1] + 2 * padding[1])/stride[1] + 1;

    Shape out_shape = Shape({batch_dim,channel_dim,out_h,out_w});

    Tensor out {out_shape._shapes,device,0.0f,true};

    device->math->max_pooling2D(
        input.data(),
        input.shape,
        this->stride,
        this->window_shape,
        out.data(),
        out.shape,
        this->padding
    );

    return out;
}

void MaxPooling::update_weights(Tensor& input){
    throw std::logic_error("Not implemented");
}

Tensor AvgPooling::forward(Tensor& input){
    
    auto device = input.getDevice();

    std::size_t batch_dim = input.shape[0];
    std::size_t channel_dim = input.shape[1];
    std::size_t out_h = (input.shape[2] - this->window_shape[0] + 2 * padding[0])/stride[0] + 1;
    std::size_t out_w = (input.shape[3] - this->window_shape[1] + 2 * padding[1])/stride[1] + 1;

    Shape out_shape = Shape({batch_dim,channel_dim,out_h,out_w});

    Tensor out {out_shape._shapes,device,0.0f,true};

    device->math->avg_pooling2D(
        input.data(),
        input.shape,
        this->stride,
        this->window_shape,
        out.data(),
        out.shape,
        this->padding
    );

    return out;
}

void AvgPooling::update_weights(Tensor& input){
    throw std::logic_error("Not implemented");
}
