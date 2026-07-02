#include <hardware/GpuDevice.cuh>
#include <stdexcept>
#include <cstdint>
#include <cstring>
#include <chrono>

#ifdef __USE_CUDA__
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

void* GpuDevice::allocate(size_t _n_bytes){
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, _n_bytes);
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
    uint32_t asRaw = 0; 
    std::memcpy(&asRaw, &value, sizeof(float));
    
    CUresult err = cuMemsetD32(reinterpret_cast<CUdeviceptr>(dst), asRaw, size / sizeof(float));
    if(err != CUDA_SUCCESS){
        throw std::runtime_error("Failed to set GPU memory");
    }
}

namespace{
    __global__ void rand_fill_kernel(
        float* dst,
        std::size_t n,
        unsigned long long seed
    ){
        size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n) return;
        curandState state;
        curand_init(seed,idx,0,&state);
        dst[idx] = curand_uniform(&state);
    }
};


/*Set `dst` memory region holding `_n` floats with a random float value from [0,1]. 
`dst` is interpreted always as a float region of memory*/
void GpuDevice::memset_rand(void* dst, size_t _n){
    float* f_dst = static_cast<float*>(dst);
    constexpr std::size_t block_size = 256;
    std::size_t grid_size = (_n + block_size - 1) / block_size;
    unsigned long long seed = (unsigned long long)std::chrono::system_clock::now()
                                .time_since_epoch().count();
    rand_fill_kernel<<<grid_size,block_size>>>(
        f_dst,
        _n,
        seed
    );

    auto result = cudaDeviceSynchronize();
    if(result != cudaSuccess){
        throw std::runtime_error("Error: Initializing memory with random values failed");
    }
    return;
}

void GpuDevice::make_contiguous(
    void* dst, 
    const void* src, 
    const std::vector<std::size_t> shapes,
    const std::vector<std::size_t> faulty_strides
){

}

DeviceType GpuDevice::get_type() const {
    return DeviceType::GPU;
}

std::string GpuDevice::get_name() const {
    return "GPU";
}

#endif //__USE_CUDA__