#include <cnn/tensor.hpp>
#include <types/memory.cuh>

namespace GPU{

    Tensor add(Tensor& a, Tensor& b);
    Tensor scalarMultiply(Tensor& a);
    Tensor multipy(Tensor& a, Tensor& b);

    CudaMemPtr allocate_memory(const float* ptr, size_t size);
    void free_memory(const CudaMemPtr&);

};