#pragma once

#include <core/Tensor.hpp>
#include <data/Dataset.hpp>

#include <cstddef>
#include <filesystem>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

class DataLoader {
public:

    struct Batch {
        Tensor images;
        std::vector<std::size_t> labels;
        std::vector<std::filesystem::path> paths;
        std::vector<std::string> class_names;
        std::vector<std::size_t> indices;
    };

    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Batch;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = Batch;

        Iterator(const DataLoader* loader, std::size_t batch_index);

        Batch operator*() const;
        Iterator& operator++();

        bool operator==(const Iterator& other) const;
        bool operator!=(const Iterator& other) const;

    private:
        const DataLoader* _loader;
        std::size_t _batch_index;
    };

    explicit DataLoader(
        const Dataset& dataset,
        std::size_t batch_size,
        bool shuffle = false,
        bool drop_last = false,
        unsigned int seed = 0
    );

    const Dataset& dataset() const { return _dataset; }
    std::size_t batch_size() const { return _batch_size; }

    bool shuffle() const { return _shuffle; }
    bool drop_last() const { return _drop_last; }

    std::size_t size() const { return _dataset.size(); }
    bool empty() const { return _dataset.empty(); }
    std::size_t num_batches() const;
    const std::vector<std::size_t>& indices() const { return _indices; }

    void reset();

    Batch get_batch(std::size_t batch_index) const;
    Iterator begin() const;
    Iterator end() const;

private:
    const Dataset& _dataset;
    std::size_t _batch_size;

    bool _shuffle;
    bool _drop_last;
    unsigned int _seed;
    std::vector<std::size_t> _indices;

    void _build_indices();
    void _shuffle_indices();
    Tensor _build_image_batch(const std::vector<std::size_t>& batch_indices) const;
};