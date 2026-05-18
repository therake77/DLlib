#include <hardware/Device.hpp>

class CpuDevice : public Device {
public:
    ~CpuDevice() override = default;

    void* allocate(size_t size) override;
    void free(void* ptr) override;
    
    void copy_to_host(void* host_dst, const void* src, size_t size) override;
    void copy_to_device(void* dst, const void* host_src, size_t size) override;
    void copy_device_to_device(void* dst, const void* src, size_t size) override;

    void memset(void* dst, float value, size_t size) override;

    DeviceType get_type() const override;
    std::string get_name() const override;
};