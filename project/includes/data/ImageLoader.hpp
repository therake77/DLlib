#pragma once

#include <filesystem>
#include <memory>
#include <vector>
#include <core/Tensor.hpp>

class ImageLoader {
public:
    explicit ImageLoader(std::shared_ptr<Device> dev = DeviceManager::get_cpu_device());

    Tensor load_image(const std::filesystem::path& path, bool grayscale = false, bool normalize = true) const;
    Tensor load_folder(const std::filesystem::path& folder, bool grayscale = false, bool normalize = true) const;

    const std::shared_ptr<Device>& device() const { return _dev; }

private:
    std::shared_ptr<Device> _dev;

    static bool _is_supported_extension(const std::filesystem::path& path);
    static Tensor _load_image_cpu(const std::filesystem::path& path, bool grayscale, bool normalize);
};