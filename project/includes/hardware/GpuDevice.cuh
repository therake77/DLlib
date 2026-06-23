#include <hardware/Device.hpp>
#include <math/GpuMath.cuh>
#ifdef __USE_CUDA__

class GpuDevice : public Device {
public:

    ~GpuDevice() override = default;
    
    GpuDevice(){ this->math = std::make_unique<GpuMath>(); }

    void* allocate(size_t size) override;
    void free(void* ptr) override;
    
    void copy_to_host(void* host_dst, const void* src, size_t size) override;
    void copy_to_device(void* dst, const void* host_src, size_t size) override;
    void copy_device_to_device(void* dst, const void* src, size_t size) override;

    void memset(void* dst, float value, size_t size) override;
    void memset_rand(void*dst, size_t size) override;

    DeviceType get_type() const override;
    std::string get_name() const override;
};

#endif // __USE_CUDA__