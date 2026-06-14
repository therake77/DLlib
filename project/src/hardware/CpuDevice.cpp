#include <hardware/CpuDevice.hpp>
#include <stdexcept>
#include <cstring>
#include <random>


void* CpuDevice::allocate(size_t _n_bytes){
    return std::malloc(_n_bytes);
}

void CpuDevice::free(void* ptr){
    std::free(ptr);
}

void CpuDevice::copy_to_host(void* host_dst, const void* src, size_t size){
    std::memcpy(host_dst, src, size);
}

void CpuDevice::copy_to_device(void* dst, const void* host_src, size_t size){
    std::memcpy(dst, host_src, size);
}

void CpuDevice::copy_device_to_device(void* dst, const void* src, size_t size){
    std::memcpy(dst, src, size);
}

/*Set `dst` memory region holding `_n` floats with a fixed float value. 
`dst` is interpreted always as a float region of memory*/
void CpuDevice::memset(void* dst, float value, size_t _n){
    std::fill_n(static_cast<float*>(dst), _n,value);
}

/*Set `dst` memory region holding `_n` floats with a random float value from [0,1]. 
`dst` is interpreted always as a float region of memory*/
void CpuDevice::memset_rand(void* dst, size_t _n){
    float* f_dst = static_cast<float*>(dst);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f,1.0f);

    for(size_t i = 0 ; i < _n; i++){
        f_dst[i] = dist(gen); 
    }
}

DeviceType CpuDevice::get_type() const {
    return DeviceType::CPU;
}

std::string CpuDevice::get_name() const {
    return "CPU";
}