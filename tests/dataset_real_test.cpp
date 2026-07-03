#include <data/Dataset.hpp>
#include <hardware/DeviceManager.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <limits>
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

        Dataset dataset(
            dataset_root,
            DeviceManager::get_cpu_device(),
            false,  // RGB
            true,   // normalize
            true    // recursive
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
                "Loaded real image should have 3 channels"
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
                    << " images\n";
            }
        }

        std::cout
            << "\nImage size statistics:\n";

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

        std::cout
            << "\nReal dataset lazy-loading test passed"
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