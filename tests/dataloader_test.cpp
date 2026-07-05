#include <data/Dataset.hpp>
#include <data/DataLoader.hpp>
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

    write_pgm(root / "alpha" / "a0.pgm", 2, 2, {
        0, 10,
        20, 30
    });

    write_pgm(root / "alpha" / "a1.pgm", 2, 2, {
        40, 50,
        60, 70
    });

    write_pgm(root / "beta" / "b0.pgm", 2, 2, {
        80, 90,
        100, 110
    });

    write_pgm(root / "beta" / "b1.pgm", 2, 2, {
        120, 130,
        140, 150
    });

    write_pgm(root / "beta" / "b2.pgm", 2, 2, {
        160, 170,
        180, 190
    });
}

int main() {
    try {
        fs::path root =
            fs::temp_directory_path() / "llcnn_dataloader_test_tmp";

        create_test_dataset(root);

        Dataset dataset(
            root,
            DeviceManager::get_cpu_device(),
            true,   // grayscale
            true,   // normalize
            false   // recursive
        );

        require(
            dataset.size() == 5,
            "Dataset size should be 5"
        );

        require(
            dataset.num_classes() == 2,
            "Dataset should have 2 classes"
        );

        require(
            dataset.class_names()[0] == "alpha",
            "First class should be alpha"
        );

        require(
            dataset.class_names()[1] == "beta",
            "Second class should be beta"
        );

        DataLoader loader(
            dataset,
            2,      // batch size
            false,  // shuffle
            false,  // drop_last
            123
        );

        require(
            loader.num_batches() == 3,
            "DataLoader should create 3 batches"
        );

        DataLoader::Batch batch0 =
            loader.get_batch(0);

        require(
            batch0.images.shape.n_dim == 4,
            "Batch tensor should be 4D"
        );

        require(
            batch0.images.shape[0] == 2,
            "First batch size should be 2"
        );

        require(
            batch0.images.shape[1] == 1,
            "Batch channels should be 1"
        );

        require(
            batch0.images.shape[2] == 2,
            "Batch height should be 2"
        );

        require(
            batch0.images.shape[3] == 2,
            "Batch width should be 2"
        );

        require(
            batch0.labels.size() == 2,
            "Batch labels size should be 2"
        );

        require(
            batch0.paths.size() == 2,
            "Batch paths size should be 2"
        );

        require(
            batch0.class_names.size() == 2,
            "Batch class_names size should be 2"
        );

        require(
            batch0.indices.size() == 2,
            "Batch indices size should be 2"
        );

        require(
            batch0.labels[0] == 0,
            "First sample label should be 0"
        );

        require(
            batch0.labels[1] == 0,
            "Second sample label should be 0"
        );

        const float* batch_data =
            batch0.images.data();

        check_close(batch_data[0], 0.0f / 255.0f);
        check_close(batch_data[1], 10.0f / 255.0f);
        check_close(batch_data[2], 20.0f / 255.0f);
        check_close(batch_data[3], 30.0f / 255.0f);

        DataLoader::Batch last_batch =
            loader.get_batch(2);

        require(
            last_batch.images.shape[0] == 1,
            "Last batch should have size 1"
        );

        DataLoader drop_loader(
            dataset,
            2,
            false,
            true,
            123
        );

        require(
            drop_loader.num_batches() == 2,
            "Drop-last DataLoader should create 2 batches"
        );

        std::size_t iterated_batches = 0;
        std::size_t iterated_samples = 0;

        for(auto batch : loader) {
            iterated_batches++;
            iterated_samples += batch.labels.size();

            require(
                batch.images.shape.n_dim == 4,
                "Iterator batch images should be 4D"
            );

            require(
                batch.labels.size() == batch.paths.size(),
                "Labels and paths should have same size"
            );
        }

        require(
            iterated_batches == 3,
            "Iterator should visit 3 batches"
        );

        require(
            iterated_samples == 5,
            "Iterator should visit all samples"
        );

        fs::remove_all(root);

        std::cout
            << "DataLoader test passed"
            << std::endl;

        return 0;
    }
    catch(const std::exception& e) {
        std::cerr
            << "DataLoader test failed: "
            << e.what()
            << std::endl;

        return 1;
    }
}