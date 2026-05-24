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
    const size_t& size = n_elem;

    Tensor(std::initializer_list<size_t>, std::shared_ptr<Device> = DeviceManager::get_cpu_device());
    Tensor(std::vector<size_t> shape, std::shared_ptr<Device> = DeviceManager::get_cpu_device(), float _fill = 0.0f);
    Tensor(Tensor&, std::shared_ptr<Device> = DeviceManager::get_cpu_device());
    float* data();
    void to(std::shared_ptr<Device>);
    const inline std::vector<size_t> getShape() const { return this->shape; }
    const inline std::shared_ptr<Device>& getDevice() const { return this->device; }
};