#pragma once

#include <memory>
#include <vector>
#include <initializer_list>
#include <hardware/Device.hpp>
#include <hardware/DeviceManager.hpp>
#include <functional>

class Tensor{
private:
    std::shared_ptr<Device> device;
    std::vector<size_t> shape;
    std::unique_ptr<void, std::function<void(void*)>> cpu_data;
    std::unique_ptr<void, std::function<void(void*)>> gpu_data;
    size_t n_elem;

public:
    Tensor(std::initializer_list<size_t>, std::shared_ptr<Device> = DeviceManager::get_cpu_device());
    Tensor(Tensor&, std::shared_ptr<Device> = DeviceManager::get_cpu_device());
    const float* data() const;
    void to(std::shared_ptr<Device>);
};