#pragma once

#include <core/Tensor.hpp>
#include <data/ImageLoader.hpp>
#include <data/transforms/Transform.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

class Dataset {
public:
    struct Sample {
        std::filesystem::path path;
        std::size_t label;
        std::string class_name;
    };

    explicit Dataset(
        const std::filesystem::path& root,
        std::shared_ptr<Device> dev = DeviceManager::get_cpu_device(),
        bool grayscale = false,
        bool normalize = true,
        bool recursive = false,
        std::shared_ptr<Transform> transform = nullptr
    );

    void reload();

    Tensor load(std::size_t index) const;
    Tensor load_cpu(std::size_t index) const;

    const Sample& operator[](std::size_t index) const;

    const std::filesystem::path& root() const {
        return _root;
    }

    const std::shared_ptr<Device>& device() const {
        return _dev;
    }

    std::size_t size() const {
        return _samples.size();
    }

    bool empty() const {
        return _samples.empty();
    }

    std::size_t num_classes() const {
        return _class_names.size();
    }

    const std::vector<std::string>& class_names() const {
        return _class_names;
    }

    const std::vector<Sample>& samples() const {
        return _samples;
    }

private:
    std::filesystem::path _root;
    std::shared_ptr<Device> _dev;
    bool _grayscale;
    bool _normalize;
    bool _recursive;
    std::shared_ptr<Transform> _transform;

    ImageLoader _loader;

    std::vector<Sample> _samples;
    std::vector<std::string> _class_names;

    static std::vector<std::filesystem::path> _collect_images(
        const std::filesystem::path& folder,
        bool recursive
    );
};
