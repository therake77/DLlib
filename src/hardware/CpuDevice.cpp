#include <hardware/CpuDevice.hpp>
#include <stdexcept>
#include <cstring>



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

void CpuDevice::memset(void* dst, float value, size_t _n){
    std::fill_n(static_cast<float*>(dst), _n,value);
}

DeviceType CpuDevice::get_type() const {
    return DeviceType::CPU;
}

std::string CpuDevice::get_name() const {
    return "CPU";
}