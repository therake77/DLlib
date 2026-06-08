#include <math/GpuMath.cuh>

#ifdef __USE_CUDA__

//Aquí irian los kernels de cuda, los cuales no voy a hacer por el momento
void GpuMath::add(const float* a, const float* b, float* out, size_t size) { return; }
void GpuMath::sub(const float* a, const float* b, float* out, size_t size) { return; }
void GpuMath::scale(const float* a, float* out, float _n, size_t size) { return; }
void GpuMath::multiply(const float* a, const float* b, float* out, size_t size) { return; }
void GpuMath::matmul(
    const float* a , Shape a_shape,  
    const float* b , Shape b_shape,
    float* out, Shape out_shape
) { return; }
void GpuMath::conv2D(
    const float* in, Shape in_shape,
    const float* ker, Shape ker_shape,
    float out, Shape out_shape,
    size_t stride, size_t padding
) { return; }
void GpuMath::conv3D(
    const float* in, Shape in_shape,
    const float* ker, Shape ker_shape,
    float out, Shape out_shape,
    size_t stride, size_t padding
) { return; }
void GpuMath::reLU(const float* a, float* out, size_t n_elements) { return; }
void GpuMath::sigmoid(const float* a, float* out, size_t n_elements) { return; }
void GpuMath::tanh(const float* a, float* out, size_t n_elements) { return; }
void GpuMath::softmax(const float* a, float* out, size_t n_elements) { return; }
void GpuMath::sum(
    const float* a, Shape a_shape, 
    float* out, Shape out_shape ,
    size_t axis
) { return; }



#endif //__USE_CUDA__