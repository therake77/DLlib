#include <kernel/CpuMath.hpp>

void CpuMath::add(float* a, float* b, size_t size, float* out){
    for(size_t i = 0; i < size; i++){
        out[i] = a[i] + b[i];
    }
}

void CpuMath::sub(float* a, float* b, size_t size, float* out){
    for(size_t i = 0; i < size; i++){
        out[i] = a[i] - b[i];
    }
}

void CpuMath::scale(float* a, size_t size, float _n, float* out){
    for(size_t i = 0; i < size; i++){
        out[i] = _n*a[i];
    }
}

void CpuMath::multiply(float* a, float* b, size_t size, float* out){
    for(size_t i = 0; i < size; i++){
        out[i] = a[i] * b[i];
    }
}