#pragma once
#include <memory>
#include <hardware/Device.hpp>

class DeviceManager {
public:
    static std::shared_ptr<Device> get_cpu_device();
    static std::shared_ptr<Device> get_gpu_device();
};