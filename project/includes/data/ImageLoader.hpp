#pragma once

#include <core/Tensor.hpp>

#include <filesystem>
#include <memory>

class ImageLoader {
public:
    explicit ImageLoader(std::shared_ptr<Device> dev = DeviceManager::get_cpu_device());

    Tensor load_image(const std::filesystem::path& path, bool grayscale = false, bool normalize = true) const;
    Tensor load_image_cpu(const std::filesystem::path& path, bool grayscale = false, bool normalize = true) const;

    const std::shared_ptr<Device>& device() const { return _dev; }
    static bool is_supported_extension(const std::filesystem::path& path);

private:
    std::shared_ptr<Device> _dev;

    static Tensor _load_image_cpu_impl(const std::filesystem::path& path, bool grayscale, bool normalize);
};