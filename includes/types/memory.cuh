#pragma once
#include <memory>
#include <cuda_runtime.h>

using Cuda_ptr_t = std::unique_ptr<float, decltype(&cudaFree)>;

struct CudaMemPtr{
        Cuda_ptr_t ptr{nullptr, &cudaFree};
        size_t size;

        CudaMemPtr() = default;
        CudaMemPtr(const CudaMemPtr&) = delete;
        CudaMemPtr& operator=(const CudaMemPtr&) = delete;
        CudaMemPtr(CudaMemPtr&&) noexcept = default;
        CudaMemPtr& operator=(CudaMemPtr&&) noexcept = default;
};