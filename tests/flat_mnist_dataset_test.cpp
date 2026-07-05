#include <data/FlatDataset.hpp>
#include <hardware/DeviceManager.hpp>

#include <cstddef>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

static void require(bool condition, const char* message) {
    if(!condition) {
        throw std::runtime_error(message);
    }
}

static void print_shape(const Tensor& tensor) {
    std::cout
        << "("
        << tensor.shape[0]
        << ", "
        << tensor.shape[1];

    if(tensor.shape.n_dim > 2) {
        std::cout
            << ", "
            << tensor.shape[2]
            << ", "
            << tensor.shape[3];
    }

    std::cout << ")";
}

int main() {
    try {
        fs::path root = "datasets/mnist/flat";
        fs::path train_images = root / "train_images_float32.bin";
        fs::path train_labels = root / "train_labels_uint8.bin";

        if(!fs::exists(train_images) || !fs::exists(train_labels)) {
            std::cout
                << "Flat MNIST files not found in "
                << root
                << ".\nRun: python scripts/export_mnist_flat.py"
                << std::endl;

            return 0;
        }

        FlatDataset train(
            train_images,
            train_labels,
            60000,
            1,
            28,
            28,
            10,
            DeviceManager::get_cpu_device()
        );

        std::cout
            << "\n========== FLAT MNIST DATASET TEST ==========\n";

        std::cout
            << "Images file: "
            << train.images_path()
            << '\n';

        std::cout
            << "Labels file: "
            << train.labels_path()
            << '\n';

        std::cout
            << "Samples: "
            << train.size()
            << '\n';

        std::cout
            << "Image shape metadata: "
            << train.channels()
            << "x"
            << train.height()
            << "x"
            << train.width()
            << '\n';

        std::cout
            << "Classes: "
            << train.num_classes()
            << '\n';

        require(train.size() == 60000, "MNIST train split should contain 60000 samples");
        require(train.channels() == 1, "MNIST should have one channel");
        require(train.height() == 28, "MNIST height should be 28");
        require(train.width() == 28, "MNIST width should be 28");
        require(train.num_classes() == 10, "MNIST should contain 10 classes");

        Tensor image(
            {1, 1, 28, 28},
            DeviceManager::get_cpu_device(),
            0.0f,
            true
        );
        train.load_cpu(0, image);

        std::uint8_t label = train.label(0);

        std::cout
            << "\nSingle sample:\n";

        std::cout
            << "  train.load_cpu(0): ";
        print_shape(image);
        std::cout
            << " = (N, C, H, W)\n";

        std::cout
            << "  train.label(0): "
            << static_cast<int>(label)
            << '\n';

        require(
            image.shape.n_dim == 4 &&
            image.shape[0] == 1 &&
            image.shape[1] == 1 &&
            image.shape[2] == 28 &&
            image.shape[3] == 28,
            "Single flat MNIST image should have shape (1, 1, 28, 28)"
        );

        require(label < 10, "MNIST label should be in range 0..9");

        const float* image_data = image.data();
        for(std::size_t i = 0; i < image.shape.n_elements; ++i) {
            require(
                image_data[i] >= 0.0f && image_data[i] <= 1.0f,
                "Flat MNIST image values should already be normalized to 0..1"
            );
        }

        float* reusable_image_ptr = image.data();
        train.load_cpu(1, image);
        require(
            image.data() == reusable_image_ptr,
            "Output-param single image should reuse memory when shape does not change"
        );

        Tensor batch(
            {8, 1, 28, 28},
            DeviceManager::get_cpu_device(),
            0.0f,
            true
        );
        train.load_batch_cpu(0, 8, batch);

        Tensor one_hot = train.labels_one_hot_cpu(0, 8);

        std::cout
            << "\nBatch:\n";

        std::cout
            << "  train.load_batch_cpu(0, 8): ";
        print_shape(batch);
        std::cout
            << " = (N, C, H, W)\n";

        std::cout
            << "  train.labels_one_hot_cpu(0, 8): ";
        print_shape(one_hot);
        std::cout
            << " = (N, classes)\n";

        std::cout
            << "  first 8 labels: ";
        for(std::size_t i = 0; i < 8; ++i) {
            std::cout
                << static_cast<int>(train.label(i))
                << (i + 1 == 8 ? '\n' : ' ');
        }

        require(
            batch.shape.n_dim == 4 &&
            batch.shape[0] == 8 &&
            batch.shape[1] == 1 &&
            batch.shape[2] == 28 &&
            batch.shape[3] == 28,
            "Flat MNIST batch should have shape (8, 1, 28, 28)"
        );

        float* reusable_batch_ptr = batch.data();
        train.load_batch_cpu(8, 8, batch);
        require(
            batch.data() == reusable_batch_ptr,
            "Output-param batch should reuse memory when shape does not change"
        );

        require(
            one_hot.shape.n_dim == 2 &&
            one_hot.shape[0] == 8 &&
            one_hot.shape[1] == 10,
            "Flat MNIST one-hot labels should have shape (8, 10)"
        );

        const float* one_hot_data = one_hot.data();
        for(std::size_t n = 0; n < 8; ++n) {
            float sum = 0.0f;
            for(std::size_t c = 0; c < 10; ++c) {
                sum += one_hot_data[n * 10 + c];
            }

            require(sum == 1.0f, "Each one-hot label should contain exactly one active class");
            require(
                one_hot_data[n * 10 + static_cast<std::size_t>(train.label(n))] == 1.0f,
                "One-hot active class should match the integer label"
            );
        }

        std::cout
            << "\nFlat MNIST dataset test passed"
            << std::endl;

        return 0;
    }
    catch(const std::exception& e) {
        std::cerr
            << "Flat MNIST dataset test failed: "
            << e.what()
            << std::endl;

        return 1;
    }
}
