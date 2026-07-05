#include <data/FlatDataset.hpp>

#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

FlatDataset::FlatDataset(
    const std::filesystem::path& images_path,
    const std::filesystem::path& labels_path,
    std::size_t count,
    std::size_t channels,
    std::size_t height,
    std::size_t width,
    std::size_t num_classes,
    std::shared_ptr<Device> dev
)
    : _images_path(images_path),
      _labels_path(labels_path),
      _dev(std::move(dev)),
      _count(count),
      _channels(channels),
      _height(height),
      _width(width),
      _num_classes(num_classes) {
    if(!_dev) {
        throw std::runtime_error("Error: Invalid device");
    }

    if(_count == 0 || _channels == 0 || _height == 0 || _width == 0 || _num_classes == 0) {
        throw std::runtime_error("Error: Invalid flat dataset metadata");
    }

    if(_num_classes > static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max()) + 1) {
        throw std::runtime_error("Error: FlatDataset labels use uint8, so num_classes cannot exceed 256");
    }

    _load_files();
}

std::size_t FlatDataset::_sample_elements() const {
    return _channels * _height * _width;
}

void FlatDataset::_load_files() {
    if(!std::filesystem::exists(_images_path) || !std::filesystem::is_regular_file(_images_path)) {
        throw std::runtime_error("Error: FlatDataset images file does not exist");
    }

    if(!std::filesystem::exists(_labels_path) || !std::filesystem::is_regular_file(_labels_path)) {
        throw std::runtime_error("Error: FlatDataset labels file does not exist");
    }

    std::size_t image_values = _count * _sample_elements();
    std::uintmax_t expected_image_bytes =
        static_cast<std::uintmax_t>(image_values) *
        static_cast<std::uintmax_t>(sizeof(float));
    std::uintmax_t expected_label_bytes =
        static_cast<std::uintmax_t>(_count) *
        static_cast<std::uintmax_t>(sizeof(std::uint8_t));

    if(std::filesystem::file_size(_images_path) != expected_image_bytes) {
        throw std::runtime_error("Error: FlatDataset images file size does not match metadata");
    }

    if(std::filesystem::file_size(_labels_path) != expected_label_bytes) {
        throw std::runtime_error("Error: FlatDataset labels file size does not match metadata");
    }

    _images.resize(image_values);
    _labels.resize(_count);

    std::ifstream images_file(_images_path, std::ios::binary);
    if(!images_file) {
        throw std::runtime_error("Error: Could not open FlatDataset images file");
    }

    std::ifstream labels_file(_labels_path, std::ios::binary);
    if(!labels_file) {
        throw std::runtime_error("Error: Could not open FlatDataset labels file");
    }

    images_file.read(
        reinterpret_cast<char*>(_images.data()),
        static_cast<std::streamsize>(_images.size() * sizeof(float))
    );

    labels_file.read(
        reinterpret_cast<char*>(_labels.data()),
        static_cast<std::streamsize>(_labels.size() * sizeof(std::uint8_t))
    );

    if(!images_file || !labels_file) {
        throw std::runtime_error("Error: Failed to read FlatDataset files");
    }

    for(std::uint8_t value : _labels) {
        if(static_cast<std::size_t>(value) >= _num_classes) {
            throw std::runtime_error("Error: FlatDataset label outside expected class range");
        }
    }
}

void FlatDataset::_check_index(std::size_t index) const {
    if(index >= _count) {
        throw std::runtime_error("Error: FlatDataset index out of bounds");
    }
}

std::size_t FlatDataset::_checked_batch_size(std::size_t start, std::size_t batch_size) const {
    if(batch_size == 0) {
        throw std::runtime_error("Error: FlatDataset batch size must be greater than zero");
    }

    if(start >= _count) {
        throw std::runtime_error("Error: FlatDataset batch start out of bounds");
    }

    std::size_t remaining = _count - start;
    return batch_size < remaining ? batch_size : remaining;
}

Tensor FlatDataset::load_cpu(std::size_t index) const {
    _check_index(index);

    std::size_t sample_elements = _sample_elements();
    const float* src = _images.data() + index * sample_elements;

    return Tensor(
        {1, _channels, _height, _width},
        src,
        DeviceManager::get_cpu_device()
    );
}



Tensor FlatDataset::load(std::size_t index) const {
    Tensor image = load_cpu(index);
    if(_dev->get_type() != DeviceType::CPU) {
        image.to(_dev);
    }
    return image;
}

Tensor FlatDataset::load_batch_cpu(std::size_t start, std::size_t batch_size) const {
    std::size_t actual_batch_size = _checked_batch_size(start, batch_size);

    std::size_t sample_elements = _sample_elements();
    const float* src = _images.data() + start * sample_elements;

    return Tensor(
        {actual_batch_size, _channels, _height, _width},
        src,
        DeviceManager::get_cpu_device()
    );
}

Tensor FlatDataset::load_batch(std::size_t start, std::size_t batch_size) const {
    Tensor batch = load_batch_cpu(start, batch_size);
    if(_dev->get_type() != DeviceType::CPU) {
        batch.to(_dev);
    }
    return batch;
}

Tensor FlatDataset::labels_one_hot_cpu(std::size_t start, std::size_t batch_size) const {
    std::size_t actual_batch_size = _checked_batch_size(start, batch_size);

    Tensor labels(
        {actual_batch_size, _num_classes, 0, 0},
        DeviceManager::get_cpu_device(),
        0.0f,
        true
    );

    float* dst = labels.data();

    for(std::size_t n = 0; n < actual_batch_size; ++n) {
        std::size_t label_index = static_cast<std::size_t>(_labels[start + n]);
        dst[n * _num_classes + label_index] = 1.0f;
    }

    return labels;
}

Tensor FlatDataset::labels_one_hot(std::size_t start, std::size_t batch_size) const {
    Tensor labels = labels_one_hot_cpu(start, batch_size);
    if(_dev->get_type() != DeviceType::CPU) {
        labels.to(_dev);
    }
    return labels;
}

std::uint8_t FlatDataset::label(std::size_t index) const {
    _check_index(index);
    return _labels[index];
}
