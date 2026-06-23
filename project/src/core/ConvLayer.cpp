#include <core/ConvLayer.hpp>
#include <math/TensorMath.hpp>

Conv2D::Conv2D(
    std::array<size_t, 4> kernel_shape,
    std::shared_ptr<Device> dev,
    std::array<size_t,2> _padding,
    std::array<size_t,2> _strides)
  : _kernels(kernel_shape,dev), 
    _grad(kernel_shape,dev), 
    padding(_padding),
    strides(_strides),
    _out({1,1,1,1},dev,0.0f,true)
{
    this->device = dev;
}

/*
@brief Computes the forward pass of a convolution layer
@param[in] input Input tensor - coming from upper layers
@param[out] output Preallocated output tensor
@result A new, freshly allocated tensor if no output was given, else `void`
@note It is absolutely not recommended to pass the input tensor as the output tensor, as the method will overwrite data during the operation
*/
Tensor& Conv2D::forward(Tensor& input){
    size_t batch_dim = input.shape[0];
    size_t c_out = this->kernels.shape[0];
    
    //out shape is (Batch, N_features, out_h, out_w)
    
    size_t out_h = ((input.shape[2] - kernels.shape[2] + 2 * padding[0]) / strides[0]) + 1;
    size_t out_w = ((input.shape[3] - kernels.shape[3] + 2 * padding[1]) / strides[1]) + 1;
    std::array<size_t, TENSOR_MAX_DIM> out_shape {
        batch_dim,
        c_out,
        out_h,
        out_w
    };

    //If _out is correctly allocated
    if(
        this->_out.shape._shapes != std::array<std::size_t,TENSOR_MAX_DIM>({batch_dim,c_out,out_h,out_w})
    ){
        this->_out.resize(Shape(out_shape));
    }

    TensorMath::conv2D(
        input,
        kernels,
        _out,
        strides,
        padding
    );


    return _out;
}

void Conv2D::update_weights(Tensor& input){
    TensorMath::add(kernels,input,kernels);
}
