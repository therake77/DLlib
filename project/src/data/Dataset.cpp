#include <data/Dataset.hpp>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <random>
#include <stdexcept>

namespace {
    std::string _to_lower(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }
}

Dataset::Dataset(
    const std::filesystem::path& root,
    std::shared_ptr<Device> dev,
    bool grayscale,
    bool normalize,
    bool recursive
)
    : _root(root),
      _dev(std::move(dev)),
      _grayscale(grayscale),
      _normalize(normalize),
      _recursive(recursive),
      _loader(DeviceManager::get_cpu_device()) {
    if(!_dev) {
        throw std::runtime_error("Error: Invalid device");
    }

    reload();
}

bool Dataset::_is_supported_extension(const std::filesystem::path& path) {
    std::string ext = _to_lower(path.extension().string());
    return ext == ".png"  ||
           ext == ".jpg"  ||
           ext == ".jpeg" ||
           ext == ".bmp"  ||
           ext == ".tga"  ||
           ext == ".gif"  ||
           ext == ".psd"  ||
           ext == ".pic"  ||
           ext == ".pnm"  ||
           ext == ".ppm"  ||
           ext == ".pgm";
}

std::vector<std::filesystem::path> Dataset::_collect_images(
    const std::filesystem::path& folder,
    bool recursive
) {
    std::vector<std::filesystem::path> images;

    if(recursive) {
        for(const auto& entry : std::filesystem::recursive_directory_iterator(folder)) {
            if(entry.is_regular_file() && _is_supported_extension(entry.path())) {
                images.push_back(entry.path());
            }
        }
    } else {
        for(const auto& entry : std::filesystem::directory_iterator(folder)) {
            if(entry.is_regular_file() && _is_supported_extension(entry.path())) {
                images.push_back(entry.path());
            }
        }
    }

    std::sort(images.begin(), images.end());
    return images;
}

void Dataset::reload() {
    _samples.clear();
    _class_names.clear();

    if(!std::filesystem::exists(_root)) {
        throw std::runtime_error("Error: Dataset root does not exist");
    }

    if(!std::filesystem::is_directory(_root)) {
        throw std::runtime_error("Error: Dataset root is not a directory");
    }

    std::vector<std::filesystem::path> root_images;
    std::vector<std::filesystem::path> class_dirs;

    for(const auto& entry : std::filesystem::directory_iterator(_root)) {
        if(entry.is_regular_file() && _is_supported_extension(entry.path())) {
            root_images.push_back(entry.path());
        } else if(entry.is_directory()) {
            class_dirs.push_back(entry.path());
        }
    }

    std::sort(root_images.begin(), root_images.end());
    std::sort(class_dirs.begin(), class_dirs.end());

    if(!root_images.empty()) {
        _class_names.push_back("default");

        for(const auto& image_path : root_images) {
            Tensor image = _loader.load_image(image_path, _grayscale, _normalize);
            _samples.push_back(Sample{
                std::make_shared<Tensor>(image),
                0,
                "default",
                image_path
            });
        }
    }

    for(const auto& class_dir : class_dirs) {
        std::vector<std::filesystem::path> images = _collect_images(class_dir, _recursive);
        if(images.empty()) {
            continue;
        }

        std::size_t label = _class_names.size();
        std::string class_name = class_dir.filename().string();
        _class_names.push_back(class_name);

        for(const auto& image_path : images) {
            Tensor image = _loader.load_image(image_path, _grayscale, _normalize);
            _samples.push_back(Sample{
                std::make_shared<Tensor>(image),
                label,
                class_name,
                image_path
            });
        }
    }

    if(_samples.empty()) {
        throw std::runtime_error("Error: No supported images found in dataset");
    }
}

const Dataset::Sample& Dataset::operator[](std::size_t index) const {
    if(index >= _samples.size()) {
        throw std::runtime_error("Error: Dataset index out of bounds");
    }
    return _samples[index];
}

Tensor Dataset::batch_images(const std::vector<std::size_t>& indices) const {
    if(indices.empty()) {
        throw std::runtime_error("Error: Cannot build an empty batch");
    }

    const Sample& first_sample = (*this)[indices.front()];
    const Shape& sample_shape = first_sample.image->shape;

    if(sample_shape.n_dim != 4 || sample_shape[0] != 1) {
        throw std::runtime_error("Error: Dataset samples must have shape (1, C, H, W)");
    }

    std::size_t batch_size = indices.size();
    std::size_t channels = sample_shape[1];
    std::size_t height = sample_shape[2];
    std::size_t width = sample_shape[3];
    std::size_t sample_elements = sample_shape.n_elements;

    Tensor batch(
        {batch_size, channels, height, width},
        DeviceManager::get_cpu_device(),
        0.0f,
        true
    );

    float* batch_data = batch.data();

    for(std::size_t i = 0; i < indices.size(); ++i) {
        const Sample& sample = (*this)[indices[i]];

        if(sample.image->shape != sample_shape) {
            throw std::runtime_error("Error: All samples in a batch must share the same shape");
        }

        std::memcpy(
            batch_data + i * sample_elements,
            sample.image->data(),
            sample_elements * sizeof(float)
        );
    }

    if(_dev->get_type() != DeviceType::CPU) {
        batch.to(_dev);
    }

    return batch;
}

std::pair<Tensor, std::vector<std::size_t>> Dataset::batch(std::size_t offset, std::size_t count) const {
    if(offset >= _samples.size()) {
        throw std::runtime_error("Error: Batch offset out of bounds");
    }

    std::size_t end = std::min(offset + count, _samples.size());
    std::vector<std::size_t> indices;
    indices.reserve(end - offset);

    for(std::size_t i = offset; i < end; ++i) {
        indices.push_back(i);
    }

    Tensor images = batch_images(indices);

    std::vector<std::size_t> labels;
    labels.reserve(indices.size());
    for(std::size_t i : indices) {
        labels.push_back(_samples[i].label);
    }

    return {std::move(images), std::move(labels)};
}

std::vector<std::size_t> Dataset::shuffled_indices(unsigned int seed) const {
    std::vector<std::size_t> indices(_samples.size());
    for(std::size_t i = 0; i < _samples.size(); ++i) {
        indices[i] = i;
    }

    if(seed == 0) {
        std::random_device rd;
        seed = rd();
    }

    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);
    return indices;
}