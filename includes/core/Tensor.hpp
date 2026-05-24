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
    Tensor(std::initializer_list<size_t> shape, std::shared_ptr<Device> device = DeviceManager::get_cpu_device());
    Tensor(const Tensor& other, std::shared_ptr<Device> device = DeviceManager::get_cpu_device());
    const float* data() const;
    float* data();
    void to(std::shared_ptr<Device> device);
    const std::vector<size_t>& get_shape() const;
    size_t size() const;
    std::shared_ptr<Device> get_device() const;
};