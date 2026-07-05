#include <data/DataLoader.hpp>

#include <algorithm>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>

DataLoader::Iterator::Iterator(
    const DataLoader* loader,
    std::size_t batch_index
)
    : _loader(loader),
      _batch_index(batch_index) {
}

DataLoader::Batch DataLoader::Iterator::operator*() const {
    return _loader->get_batch(_batch_index);
}

DataLoader::Iterator& DataLoader::Iterator::operator++() {
    ++_batch_index;
    return *this;
}

bool DataLoader::Iterator::operator==(const Iterator& other) const {
    return _loader == other._loader && _batch_index == other._batch_index;
}

bool DataLoader::Iterator::operator!=(const Iterator& other) const {
    return !(*this == other);
}

DataLoader::DataLoader(
    const Dataset& dataset,
    std::size_t batch_size,
    bool shuffle,
    bool drop_last,
    unsigned int seed
)
    : _dataset(dataset),
      _batch_size(batch_size),
      _shuffle(shuffle),
      _drop_last(drop_last),
      _seed(seed) {
    if(_batch_size == 0) {
        throw std::runtime_error("Error: DataLoader batch_size must be greater than zero");
    }

    if(_dataset.empty()) {
        throw std::runtime_error("Error: Cannot create DataLoader from an empty dataset");
    }

    reset();
}

void DataLoader::_build_indices() {
    _indices.resize(_dataset.size());

    std::iota(
        _indices.begin(),
        _indices.end(),
        static_cast<std::size_t>(0)
    );
}

void DataLoader::_shuffle_indices() {
    unsigned int seed = _seed;

    if(seed == 0){
        std::random_device rd;
        seed = rd();
    }

    std::mt19937 rng(seed);

    std::shuffle(
        _indices.begin(),
        _indices.end(),
        rng
    );
}

void DataLoader::reset() {
    _build_indices();

    if(_shuffle){
        _shuffle_indices();
    }
}

std::size_t DataLoader::num_batches() const {
    if(_drop_last){
        return _dataset.size() / _batch_size;
    }

    return (_dataset.size() + _batch_size - 1) / _batch_size;
}

DataLoader::Iterator DataLoader::begin() const {
    return Iterator(this, 0);
}

DataLoader::Iterator DataLoader::end() const {
    return Iterator(this, num_batches());
}

DataLoader::Batch DataLoader::get_batch(std::size_t batch_index) const{
    std::size_t total_batches = num_batches();

    if(batch_index >= total_batches) {
        throw std::runtime_error("Error: DataLoader batch index out of bounds");
    }

    std::size_t start = batch_index * _batch_size;
    std::size_t end = std::min(start + _batch_size, _indices.size());

    if(_drop_last && end - start < _batch_size) {
        throw std::runtime_error("Error: Incomplete batch requested while drop_last is enabled");
    }

    std::vector<std::size_t> batch_indices;
    batch_indices.reserve(end - start);

    for(std::size_t i = start; i < end; ++i) {
        batch_indices.push_back(_indices[i]);
    }

    Tensor images =_build_image_batch(batch_indices);

    std::vector<std::size_t> labels;
    std::vector<std::filesystem::path> paths;
    std::vector<std::string> class_names;

    labels.reserve(batch_indices.size());
    paths.reserve(batch_indices.size());
    class_names.reserve(batch_indices.size());

    for(std::size_t dataset_index : batch_indices) {
        const Dataset::Sample& sample = _dataset[dataset_index];

        labels.push_back(sample.label);
        paths.push_back(sample.path);
        class_names.push_back(sample.class_name);
    }

    return Batch{
        std::move(images),
        std::move(labels),
        std::move(paths),
        std::move(class_names),
        std::move(batch_indices)
    };
}

Tensor DataLoader::_build_image_batch(const std::vector<std::size_t>& batch_indices) const {
    if(batch_indices.empty()) {
        throw std::runtime_error("Error: Cannot build an empty batch");
    }

    Tensor first = _dataset.load_cpu(batch_indices.front());

    const Shape& sample_shape = first.shape;

    if(sample_shape.n_dim != 4 || sample_shape[0] != 1){
        throw std::runtime_error("Error: Dataset samples must have shape (1, C, H, W)");
    }

    std::size_t batch_size = batch_indices.size();
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

    std::memcpy(
        batch_data,
        first.data(),
        sample_elements * sizeof(float)
    );

    for(std::size_t i = 1; i < batch_indices.size(); ++i){
        Tensor image = _dataset.load_cpu(batch_indices[i]);

        if(image.shape != sample_shape){
            throw std::runtime_error("Error: All images in a batch must have the same shape. Use resize/crop/padding before batching.");
        }

        std::memcpy(
            batch_data + i * sample_elements,
            image.data(),
            sample_elements * sizeof(float)
        );

    }

    if(_dataset.device()->get_type() != DeviceType::CPU){
        batch.to(_dataset.device());
    }

    return batch;
}