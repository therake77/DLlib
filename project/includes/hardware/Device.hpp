#pragma once
#include <memory>
#include <string>
#include <math/Math.hpp>
#include <math/CpuMath.hpp>

#ifdef __USE_CUDA
    #include <math/GpuMath.cuh>
#endif
enum class DeviceType {
    CPU,
    GPU
};

class Device {
public:
    std::unique_ptr<Math> math;
    virtual ~Device() = default;
    Device() = default;

    virtual void* allocate(size_t size) = 0;
    virtual void free(void* ptr) = 0;
    
    virtual void copy_to_host(void* host_dst, const void* src, size_t size) = 0;
    virtual void copy_to_device(void* dst, const void* host_src, size_t size) = 0;
    virtual void copy_device_to_device(void* dst, const void* src, size_t size) = 0;

    virtual void memset(void* dst, float value, size_t size) = 0;
    
    virtual void memset_rand(void*dst, size_t size) = 0;

    virtual DeviceType get_type() const = 0;
    virtual std::string get_name() const = 0;
};


