#include <hardware/GpuDevice.cuh>
#include <stdexcept>
#include <cstdint>
#include <cstring>

#ifdef __USE_CUDA__
#include <cuda_runtime.h>


void* GpuDevice::allocate(size_t size){
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if(err != cudaSuccess){
        throw std::runtime_error("Failed to allocate GPU memory");
    }
    return ptr;
}

void GpuDevice::free(void* ptr){
    cudaFree(ptr);
}

void GpuDevice::copy_to_host(void* host_dst, const void* src, size_t size){
    cudaError_t err = cudaMemcpy(host_dst, src, size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        throw std::runtime_error("Failed to copy data from GPU to host");
    }
}

void GpuDevice::copy_to_device(void* dst, const void* host_src, size_t size){
    cudaError_t err = cudaMemcpy(dst, host_src, size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        throw std::runtime_error("Failed to copy data from host to GPU");
    }
}

void GpuDevice::copy_device_to_device(void* dst, const void* src, size_t size){
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess){
        throw std::runtime_error("Failed to copy data within GPU");
    }
}

void GpuDevice::memset(void* dst, float value, size_t size){
    uint32_t asRaw = std::memcpy(&asRaw, &value, sizeof(float));
    cudaError_t err = cudaMemsetD32(std::reinterpret_cast<CudaDevicePtr>(dst), asRaw, size / sizeof(float));
    if(err != cudaSuccess){
        throw std::runtime_error("Failed to set GPU memory");
    }
}

DeviceType GpuDevice::get_type() const {
    return DeviceType::GPU;
}

std::string GpuDevice::get_name() const {
    return "GPU";
}

#endif //__USE_CUDA__