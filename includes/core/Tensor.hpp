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
    std::vector<size_t> _shape;
    std::unique_ptr<void, std::function<void(void*)>> cpu_data;
    std::unique_ptr<void, std::function<void(void*)>> gpu_data;
    size_t _size;

public:
    const size_t& size = _size;
    const std::vector<size_t>& shape = _shape;

    Tensor(std::initializer_list<size_t>, std::shared_ptr<Device> = DeviceManager::get_cpu_device(), float _fill = 0.0f);
    Tensor(std::vector<size_t> shape, std::shared_ptr<Device> = DeviceManager::get_cpu_device(), float _fill = 0.0f);
    Tensor(Tensor&, std::shared_ptr<Device> = DeviceManager::get_cpu_device());
    float* data();
    const float* data() const;
    void to(std::shared_ptr<Device>);
    const inline std::shared_ptr<Device>& getDevice() const { return this->device; }
};