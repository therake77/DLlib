#include <math/GpuMath.cuh>

#ifdef __USE_CUDA__

//Aquí irian los kernels de cuda, los cuales no voy a hacer por el momento
void GpuMath::add(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {

}
void GpuMath::sub(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {

}
void GpuMath::scale(const float* a, float* out, float _n, Shape shape) {

}
void GpuMath::multiply(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {

}
void GpuMath::divide(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {

}
void GpuMath::sqrt(const float* a,  Shape a_shape, float* out, Shape out_shape) {

}
void GpuMath::pow(const float* a,  Shape a_shape, int _n, float* out, Shape out_shape) {

}

void GpuMath::log(const float* a,  Shape a_shape, float* out, Shape out_shape){

}


    //Linear Algebra
void GpuMath::matmul(
    const float* a , Shape a_shape,  
    const float* b , Shape b_shape,
    float* out, Shape out_shape
) {

}
    
void GpuMath::conv2D(
    const float* in, Shape in_shape,
    const float* ker, Shape ker_shape,
    float* out, Shape out_shape,
    std::array<size_t,2> stride, std::array<size_t,2> padding
) {

}

void GpuMath::im2col2d(
    const float* in, Shape in_shape,
    float* out, Shape out_shape,
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding
){

}

void GpuMath::col2im2d(
    const float* in, Shape in_shape,
    float* out, Shape out_shape,
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding
){

}

//Activation functions
void GpuMath::reLU(
    const float* a, float* out, size_t n_elements
) {

}

void GpuMath::sigmoid(const float* a, float* out, size_t n_elements) {

}
void GpuMath::tanh(const float* a, float* out, size_t n_elements) {

}
void GpuMath::softmax(
    const float* a,
    Shape a_shape, 
    float* out,
    Shape out_shape
) {

}
    
    //Structural ops and reshaping
void GpuMath::sum(
    const float* a, Shape a_shape, 
    float* out, Shape out_shape ,
    size_t axis
) {

}

    //Pooling operations
void GpuMath::max_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    float* mask,
    const Shape mask_shape,
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
) {

}

void GpuMath::inv_max_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    const float* mask,
    const Shape mask_shape,
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
) {

}

void GpuMath::avg_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
) {

}

void GpuMath::inv_avg_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
) {

}
    
void GpuMath::binary_positive_mask(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape
) {

}

void GpuMath::mean(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape,
    std::size_t axis
) {

}

void GpuMath::variance(
    const float* in,
    const Shape& in_shape,
    const float* mean,
    const Shape& mean_shape,
    float* out,
    const Shape& out_shape,
    std::size_t axis
) {

}

void GpuMath::reduce_all(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape,
    std::size_t axis
) {

}

void GpuMath::transpose(
    const float* in, const Shape& in_shape,
    float* out, const Shape& out_shape
) {

}

#endif //__USE_CUDA__