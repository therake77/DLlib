#include <data/Dataset.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>

Dataset::Dataset(
    const std::filesystem::path& root,
    std::shared_ptr<Device> dev,
    bool grayscale,
    bool normalize,
    bool recursive,
    std::shared_ptr<Transform> transform
)
    : _root(root),
      _dev(std::move(dev)),
      _grayscale(grayscale),
      _normalize(normalize),
      _recursive(recursive),
      _transform(std::move(transform)),
      _loader(DeviceManager::get_cpu_device()) {
    if(!_dev) {
        throw std::runtime_error("Error: Invalid device");
    }

    reload();
}

std::vector<std::filesystem::path> Dataset::_collect_images(
    const std::filesystem::path& folder,
    bool recursive
) {
    std::vector<std::filesystem::path> images;

    if(recursive) {
        for(const auto& entry : std::filesystem::recursive_directory_iterator(folder)) {
            if(entry.is_regular_file() && ImageLoader::is_supported_extension(entry.path())) {
                images.push_back(entry.path());
            }
        }
    } else {
        for(const auto& entry : std::filesystem::directory_iterator(folder)) {
            if(entry.is_regular_file() && ImageLoader::is_supported_extension(entry.path())) {
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
        if(entry.is_regular_file() && ImageLoader::is_supported_extension(entry.path())) {
            root_images.push_back(entry.path());
        } else if(entry.is_directory()) {
            class_dirs.push_back(entry.path());
        }
    }

    std::sort(root_images.begin(), root_images.end());
    std::sort(class_dirs.begin(), class_dirs.end());

    if(!root_images.empty()) {
        _class_names.push_back("default");

        for(const auto& image : root_images) {
            _samples.push_back({
                image,
                0,
                "default"
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

        for(const auto& image : images) {
            _samples.push_back({
                image,
                label,
                class_name
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

Tensor Dataset::load_cpu(std::size_t index) const {
    if(index >= _samples.size()) {
        throw std::runtime_error("Error: Dataset index out of bounds");
    }

    Tensor image = _loader.load_image_cpu(
        _samples[index].path,
        _grayscale,
        _normalize
    );

    if(_transform) {
        return _transform->apply(image);
    }

    return image;
}

Tensor Dataset::load(std::size_t index) const {
    Tensor image = load_cpu(index);
    if(_dev->get_type() != DeviceType::CPU) {
        image.to(_dev);
    }
    return image;
}
