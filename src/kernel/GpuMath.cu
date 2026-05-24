#include <kernel/GpuMath.cuh>

#ifdef __USE_CUDA__

//Aquí irian los kernels de cuda, los cuales no voy a hacer por el momento

void GpuMath::add(float* a, float* b, size_t size, float* out){
    return;
}

void GpuMath::sub(float* a, float* b, size_t size, float* out){
    return;
}

void GpuMath::scale(float* a, size_t size, float _n, float* out){
    return;
}

void GpuMath::multiply(float* a, float* b, size_t size, float* out){
    return;
}

void GpuMath::matmul(float* a, std::vector<size_t> a_shape, float* b, std::vector<size_t> b_shape, float* out){
    return;
}

void GpuMath::conv(float* in, std::vector<size_t> in_shape, float* ker, std::vector<size_t> ker_shape, float* out, size_t stride, size_t padding){
    return;
}

//Activation functions
void GpuMath::reLU(float* a, size_t size, float* out){
    return;
}

void GpuMath::sigmoid(float* a, size_t size, float* out){
    return;
}

void GpuMath::tanh(float* a, size_t size, float* out){
    return;
}

void GpuMath::softmax(float* a, size_t size, float* out){
    return;
}

//Structural ops and reshaping
void GpuMath::sum(float* a, std::vector<size_t> shape, size_t axis, float* out){
    return;
}

//TODO(Do we really need a reshape at this level?)
void GpuMath::reshape(float* a, std::vector<size_t> shape, std::vector<size_t> new_shape, float* out){
    return;
}

void GpuMath::transpose(float* a, std::vector<size_t> shape, float* out){
    return;
}

#endif //__USE_CUDA__