#pragma once

#include <core/Tensor.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

class FlatDataset {
public:
    explicit FlatDataset(
        const std::filesystem::path& images_path,
        const std::filesystem::path& labels_path,
        std::size_t count,
        std::size_t channels,
        std::size_t height,
        std::size_t width,
        std::size_t num_classes,
        std::shared_ptr<Device> dev = DeviceManager::get_cpu_device()
    );

    Tensor load_cpu(std::size_t index) const;
    void load_cpu(std::size_t index, Tensor& out) const;
    Tensor load(std::size_t index) const;

    Tensor load_batch_cpu(std::size_t start, std::size_t batch_size) const;
    void load_batch_cpu(std::size_t start, std::size_t batch_size, Tensor& out) const;
    Tensor load_batch(std::size_t start, std::size_t batch_size) const;

    Tensor labels_one_hot_cpu(std::size_t start, std::size_t batch_size) const;
    Tensor labels_one_hot(std::size_t start, std::size_t batch_size) const;

    std::uint8_t label(std::size_t index) const;

    std::size_t size() const { return _count; }
    bool empty() const { return _count == 0; }
    std::size_t channels() const { return _channels; }
    std::size_t height() const { return _height; }
    std::size_t width() const { return _width; }
    std::size_t num_classes() const { return _num_classes; }

    const std::filesystem::path& images_path() const { return _images_path; }
    const std::filesystem::path& labels_path() const { return _labels_path; }
    const std::vector<std::uint8_t>& labels() const { return _labels; }

private:
    std::filesystem::path _images_path;
    std::filesystem::path _labels_path;
    std::shared_ptr<Device> _dev;

    std::size_t _count;
    std::size_t _channels;
    std::size_t _height;
    std::size_t _width;
    std::size_t _num_classes;

    std::vector<float> _images;
    std::vector<std::uint8_t> _labels;

    std::size_t _sample_elements() const;
    void _load_files();
    void _check_index(std::size_t index) const;
    std::size_t _checked_batch_size(std::size_t start, std::size_t batch_size) const;
};
