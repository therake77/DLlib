#include <data/Dataset.hpp>
#include <data/DataLoader.hpp>
#include <data/transforms/CenterCrop.hpp>
#include <data/transforms/Compose.hpp>
#include <data/transforms/Normalize.hpp>
#include <data/transforms/Resize.hpp>
#include <hardware/DeviceManager.hpp>

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
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
        throw std::runtime_error("Unexpected transformed tensor value");
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

static void create_variable_size_dataset(const fs::path& root) {
    fs::remove_all(root);
    fs::create_directories(root / "alpha");
    fs::create_directories(root / "beta");

    write_pgm(root / "alpha" / "a0.pgm", 2, 2, {
        10, 10,
        10, 10
    });

    write_pgm(root / "alpha" / "a1.pgm", 3, 3, {
        20, 20, 20,
        20, 20, 20,
        20, 20, 20
    });

    write_pgm(root / "beta" / "b0.pgm", 4, 2, {
        30, 30, 30, 30,
        30, 30, 30, 30
    });
}

int main() {
    try {
        fs::path root =
            fs::temp_directory_path() / "llcnn_dataset_transform_test_tmp";

        create_variable_size_dataset(root);

        auto transform = std::make_shared<Compose>(
            std::initializer_list<std::shared_ptr<Transform>>{
                std::make_shared<Resize>(4, 4),
                std::make_shared<CenterCrop>(2, 2),
                std::make_shared<Normalize>(
                    10.0f / 255.0f,
                    10.0f / 255.0f
                )
            }
        );

        Dataset dataset(
            root,
            DeviceManager::get_cpu_device(),
            true,
            true,
            false,
            transform
        );

        require(
            dataset.size() == 3,
            "Dataset size should be 3"
        );

        Tensor image = dataset.load_cpu(1);

        require(
            image.shape.n_dim == 4,
            "Transformed image should be 4D"
        );

        require(
            image.shape[0] == 1,
            "Transformed image batch dimension should be 1"
        );

        require(
            image.shape[1] == 1,
            "Transformed image channel dimension should be 1"
        );

        require(
            image.shape[2] == 2,
            "Transformed image height should be 2"
        );

        require(
            image.shape[3] == 2,
            "Transformed image width should be 2"
        );

        const float* image_data = image.data();

        for(std::size_t i = 0; i < image.shape.n_elements; ++i) {
            check_close(image_data[i], 1.0f);
        }

        DataLoader loader(
            dataset,
            3,
            false,
            false,
            123
        );

        DataLoader::Batch batch = loader.get_batch(0);

        require(
            batch.images.shape[0] == 3,
            "Batch size should be 3"
        );

        require(
            batch.images.shape[1] == 1,
            "Batch channels should be 1"
        );

        require(
            batch.images.shape[2] == 2,
            "Batch height should be 2"
        );

        require(
            batch.images.shape[3] == 2,
            "Batch width should be 2"
        );

        fs::remove_all(root);

        std::cout
            << "Dataset transform test passed"
            << std::endl;

        return 0;
    }
    catch(const std::exception& e) {
        std::cerr
            << "Dataset transform test failed: "
            << e.what()
            << std::endl;

        return 1;
    }
}
