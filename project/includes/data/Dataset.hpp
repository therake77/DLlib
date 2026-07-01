#pragma once

#include <core/Tensor.hpp>
#include <data/ImageLoader.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class Dataset {
public:
    struct Sample {
        std::shared_ptr<Tensor> image;
        std::size_t label;
        std::string class_name;
        std::filesystem::path path;
    };

    explicit Dataset(
        const std::filesystem::path& root,
        std::shared_ptr<Device> dev = DeviceManager::get_cpu_device(),
        bool grayscale = false,
        bool normalize = true,
        bool recursive = false
    );

    void reload();

    const std::filesystem::path& root() const { return _root; }
    const std::shared_ptr<Device>& device() const { return _dev; }

    std::size_t size() const { return _samples.size(); }
    bool empty() const { return _samples.empty(); }
    std::size_t num_classes() const { return _class_names.size(); }

    const std::vector<std::string>& class_names() const { return _class_names; }
    const std::vector<Sample>& samples() const { return _samples; }

    const Sample& operator[](std::size_t index) const;

    Tensor batch_images(const std::vector<std::size_t>& indices) const;
    std::pair<Tensor, std::vector<std::size_t>> batch(std::size_t offset, std::size_t count) const;
    std::vector<std::size_t> shuffled_indices(unsigned int seed = 0) const;

private:
    std::filesystem::path _root;
    std::shared_ptr<Device> _dev;
    bool _grayscale;
    bool _normalize;
    bool _recursive;

    ImageLoader _loader;
    std::vector<Sample> _samples;
    std::vector<std::string> _class_names;

    static bool _is_supported_extension(const std::filesystem::path& path);
    static std::vector<std::filesystem::path> _collect_images(
        const std::filesystem::path& folder,
        bool recursive
    );
};