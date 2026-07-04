#include <data/DataLoader.hpp>
#include <data/Dataset.hpp>
#include <data/transforms/CenterCrop.hpp>
#include <data/transforms/Compose.hpp>
#include <data/transforms/Normalize.hpp>
#include <data/transforms/Resize.hpp>
#include <hardware/DeviceManager.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

static void require(bool condition, const char* message) {
    if(!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        fs::path dataset_root =
            "datasets/perros";

        if(!fs::exists(dataset_root)) {
            std::cout
                << "Real dataset not found at "
                << dataset_root
                << ". Skipping real dataset test."
                << std::endl;

            return 0;
        }

        auto transform = std::make_shared<Compose>(
            std::initializer_list<std::shared_ptr<Transform>>{
                std::make_shared<Resize>(256, 256),
                std::make_shared<CenterCrop>(224, 224),
                std::make_shared<Normalize>(
                    std::vector<float>{0.485f, 0.456f, 0.406f},
                    std::vector<float>{0.229f, 0.224f, 0.225f}
                )
            }
        );

        Dataset dataset(
            dataset_root,
            DeviceManager::get_cpu_device(),
            false,  // RGB
            true,   // normalize
            true,   // recursive
            transform
        );

        require(
            !dataset.empty(),
            "Dataset should not be empty"
        );

        require(
            dataset.num_classes() > 0,
            "Dataset should contain at least one class"
        );

        std::cout
            << "\n========== REAL DATASET SUMMARY ==========\n";

        std::cout
            << "Dataset root: "
            << dataset.root()
            << '\n';

        std::cout
            << "Images: "
            << dataset.size()
            << '\n';

        std::cout
            << "Classes: "
            << dataset.num_classes()
            << '\n';

        std::cout
            << "Transform: Resize(256, 256) -> CenterCrop(224, 224) -> Normalize(RGB)\n";

        std::cout
            << "\nFirst classes:\n";

        std::size_t classes_to_print =
            std::min<std::size_t>(10, dataset.class_names().size());

        for(std::size_t i = 0; i < classes_to_print; ++i) {
            std::cout
                << "  ["
                << i
                << "] "
                << dataset.class_names()[i]
                << '\n';
        }

        std::size_t min_height =
            std::numeric_limits<std::size_t>::max();

        std::size_t min_width =
            std::numeric_limits<std::size_t>::max();

        std::size_t max_height = 0;
        std::size_t max_width = 0;

        std::size_t images_to_check =
            dataset.size();

        std::cout
            << "\nChecking transformed images: "
            << images_to_check
            << " / "
            << dataset.size()
            << '\n';

        for(std::size_t i = 0; i < images_to_check; ++i) {
            const Dataset::Sample& sample =
                dataset[i];

            require(
                sample.label < dataset.num_classes(),
                "Invalid sample label"
            );

            require(
                fs::exists(sample.path),
                "Sample image path does not exist"
            );

            Tensor image =
                dataset.load_cpu(i);

            const Shape& shape =
                image.shape;

            require(
                shape.n_dim == 4,
                "Loaded real image should be 4D"
            );

            require(
                shape[0] == 1,
                "Loaded real image batch dimension should be 1"
            );

            require(
                shape[1] == 3,
                "Transformed real image should have 3 channels"
            );

            require(
                shape[2] == 224,
                "Transformed real image height should be 224"
            );

            require(
                shape[3] == 224,
                "Transformed real image width should be 224"
            );

            std::size_t height =
                shape[2];

            std::size_t width =
                shape[3];

            min_height = std::min(min_height, height);
            min_width = std::min(min_width, width);

            max_height = std::max(max_height, height);
            max_width = std::max(max_width, width);

            if((i + 1) % 500 == 0) {
                std::cout
                    << "Checked "
                    << (i + 1)
                    << " / "
                    << images_to_check
                    << " transformed images\n";
            }
        }

        std::cout
            << "\nTransformed image size statistics:\n";

        std::cout
            << "Minimum height x width: "
            << min_height
            << " x "
            << min_width
            << '\n';

        std::cout
            << "Maximum height x width: "
            << max_height
            << " x "
            << max_width
            << '\n';

        require(
            dataset.size() >= 4,
            "Real dataset should contain at least 4 images for DataLoader check"
        );

        DataLoader loader(
            dataset,
            4,
            false,
            false,
            123
        );

        DataLoader::Batch batch =
            loader.get_batch(0);

        require(
            batch.images.shape.n_dim == 4,
            "Transformed real batch should be 4D"
        );

        require(
            batch.images.shape[0] == 4,
            "Transformed real batch size should be 4"
        );

        require(
            batch.images.shape[1] == 3,
            "Transformed real batch should have 3 channels"
        );

        require(
            batch.images.shape[2] == 224,
            "Transformed real batch height should be 224"
        );

        require(
            batch.images.shape[3] == 224,
            "Transformed real batch width should be 224"
        );

        require(
            batch.labels.size() == 4,
            "Transformed real batch labels size should be 4"
        );

        require(
            batch.paths.size() == 4,
            "Transformed real batch paths size should be 4"
        );

        require(
            batch.class_names.size() == 4,
            "Transformed real batch class names size should be 4"
        );

        std::cout
            << "\nReal dataset input pipeline test passed"
            << std::endl;

        return 0;
    }
    catch(const std::exception& e) {
        std::cerr
            << "\nReal dataset test failed: "
            << e.what()
            << std::endl;

        return 1;
    }
}
