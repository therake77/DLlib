#include <data/Dataset.hpp>
#include <hardware/DeviceManager.hpp>

#include <cassert>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

static void require(bool condition, const char* message) {
    if(!condition) {
        throw std::runtime_error(message);
    }
}

static void write_pgm(const fs::path& path, int width, int height, const std::vector<unsigned char>& pixels) {
    require(static_cast<int>(pixels.size()) == width * height, "PGM pixel count mismatch");

    std::ofstream file(path, std::ios::binary);
    require(file.is_open(), "Failed to create image file");

    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<const char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
    require(static_cast<bool>(file), "Failed to write image data");
}

static void create_test_dataset(const fs::path& root) {
    fs::remove_all(root);
    fs::create_directories(root / "alpha");
    fs::create_directories(root / "beta");

    write_pgm(root / "root_0.pgm", 2, 2, {
        0, 255,
        128, 64
    });

    write_pgm(root / "alpha" / "a0.pgm", 2, 2, {
        10, 20,
        30, 40
    });

    write_pgm(root / "beta" / "b0.pgm", 2, 2, {
        50, 60,
        70, 80
    });
}

static void check_close(float value, float expected, float eps = 1e-5f) {
    if(std::abs(value - expected) > eps) {
        throw std::runtime_error("Unexpected tensor value");
    }
}

int main() {
    try {
        fs::path root = fs::temp_directory_path() / "llcnn_dataset_test_tmp";
        create_test_dataset(root);

        auto device = DeviceManager::get_cpu_device();
        Dataset dataset(root, device, true, true, false);

        require(dataset.size() == 3, "Dataset size should be 3");
        require(dataset.num_classes() == 3, "Dataset should have 3 classes");
        require(dataset.class_names().size() == 3, "Class names size should be 3");
        require(dataset.class_names()[0] == "default", "First class should be default");
        require(dataset.class_names()[1] == "alpha", "Second class should be alpha");
        require(dataset.class_names()[2] == "beta", "Third class should be beta");

        const auto& sample0 = dataset[0];
        require(sample0.label == 0, "Root image label should be 0");
        require(sample0.class_name == "default", "Root image class name should be default");
        require(sample0.image != nullptr, "Sample image should not be null");
        require(sample0.image->shape.n_dim == 4, "Sample tensor should be 4D");
        require(sample0.image->shape[0] == 1, "Sample batch dimension should be 1");
        require(sample0.image->shape[1] == 1, "Sample channel dimension should be 1");
        require(sample0.image->shape[2] == 2, "Sample height should be 2");
        require(sample0.image->shape[3] == 2, "Sample width should be 2");

        const float* data = sample0.image->data();
        check_close(data[0], 0.0f);
        check_close(data[1], 1.0f);
        check_close(data[2], 128.0f / 255.0f);
        check_close(data[3], 64.0f / 255.0f);

        std::vector<std::size_t> indices = {0, 1};
        Tensor batch = dataset.batch_images(indices);
        require(batch.shape.n_dim == 4, "Batch tensor should be 4D");
        require(batch.shape[0] == 2, "Batch size should be 2");
        require(batch.shape[1] == 1, "Batch channels should be 1");
        require(batch.shape[2] == 2, "Batch height should be 2");
        require(batch.shape[3] == 2, "Batch width should be 2");

        auto [batch_tensor, labels] = dataset.batch(0, 2);
        require(batch_tensor.shape.n_dim == 4, "Batch() tensor should be 4D");
        require(batch_tensor.shape[0] == 2, "Batch() size should be 2");
        require(labels.size() == 2, "Labels size should be 2");
        require(labels[0] == 0, "First label should be 0");
        require(labels[1] == 1, "Second label should be 1");

        std::vector<std::size_t> shuffled = dataset.shuffled_indices(12345);
        require(shuffled.size() == 3, "Shuffled indices size should be 3");

        std::vector<bool> seen(3, false);
        for(std::size_t idx : shuffled) {
            require(idx < 3, "Shuffled index out of range");
            seen[idx] = true;
        }
        require(seen[0] && seen[1] && seen[2], "Shuffled indices should contain all elements");

        fs::remove_all(root);

        std::cout << "Dataset test passed" << std::endl;
        return 0;
    } catch(const std::exception& e) {
        std::cerr << "Dataset test failed: " << e.what() << std::endl;
        return 1;
    }
}