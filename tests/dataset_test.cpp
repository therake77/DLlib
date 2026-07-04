#include <data/Dataset.hpp>
#include <hardware/DeviceManager.hpp>

#include <cmath>
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

static void check_close(float value, float expected, float eps = 1e-5f) {
    if(std::fabs(value - expected) > eps) {
        throw std::runtime_error("Unexpected tensor value");
    }
}

static void write_pgm(
    const fs::path& path,
    int width,
    int height,
    const std::vector<unsigned char>& pixels
) {
    require(
        static_cast<int>(pixels.size()) == width * height,
        "PGM pixel count mismatch"
    );

    std::ofstream file(path, std::ios::binary);

    require(
        file.is_open(),
        "Failed to create image file"
    );

    file << "P5\n"
         << width
         << " "
         << height
         << "\n255\n";

    file.write(
        reinterpret_cast<const char*>(pixels.data()),
        static_cast<std::streamsize>(pixels.size())
    );

    require(
        static_cast<bool>(file),
        "Failed to write image data"
    );
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

int main() {
    try {
        fs::path root =
            fs::temp_directory_path() / "llcnn_dataset_test_tmp";

        create_test_dataset(root);

        auto device =
            DeviceManager::get_cpu_device();

        Dataset dataset(
            root,
            device,
            true,   // grayscale
            true,   // normalize
            false   // recursive
        );

        require(
            dataset.size() == 3,
            "Dataset size should be 3"
        );

        require(
            dataset.num_classes() == 3,
            "Dataset should have 3 classes"
        );

        require(
            dataset.class_names().size() == 3,
            "Class names size should be 3"
        );

        require(
            dataset.class_names()[0] == "default",
            "First class should be default"
        );

        require(
            dataset.class_names()[1] == "alpha",
            "Second class should be alpha"
        );

        require(
            dataset.class_names()[2] == "beta",
            "Third class should be beta"
        );

        const Dataset::Sample& sample0 =
            dataset[0];

        require(
            sample0.label == 0,
            "Root image label should be 0"
        );

        require(
            sample0.class_name == "default",
            "Root image class name should be default"
        );

        require(
            fs::exists(sample0.path),
            "Sample path should exist"
        );

        Tensor image0 =
            dataset.load_cpu(0);

        require(
            image0.shape.n_dim == 4,
            "Loaded image tensor should be 4D"
        );

        require(
            image0.shape[0] == 1,
            "Loaded image batch dimension should be 1"
        );

        require(
            image0.shape[1] == 1,
            "Loaded image channel dimension should be 1"
        );

        require(
            image0.shape[2] == 2,
            "Loaded image height should be 2"
        );

        require(
            image0.shape[3] == 2,
            "Loaded image width should be 2"
        );

        const float* data =
            image0.data();

        check_close(data[0], 0.0f);
        check_close(data[1], 1.0f);
        check_close(data[2], 128.0f / 255.0f);
        check_close(data[3], 64.0f / 255.0f);

        Tensor image1 =
            dataset.load(1);

        require(
            image1.shape.n_dim == 4,
            "Dataset::load should return a 4D tensor"
        );

        require(
            image1.shape[0] == 1,
            "Dataset::load image batch dimension should be 1"
        );

        require(
            image1.shape[1] == 1,
            "Dataset::load image channel dimension should be 1"
        );

        require(
            image1.shape[2] == 2,
            "Dataset::load image height should be 2"
        );

        require(
            image1.shape[3] == 2,
            "Dataset::load image width should be 2"
        );

        fs::remove_all(root);

        std::cout
            << "Dataset lazy-loading test passed"
            << std::endl;

        return 0;
    }
    catch(const std::exception& e) {
        std::cerr
            << "Dataset lazy-loading test failed: "
            << e.what()
            << std::endl;

        return 1;
    }
}