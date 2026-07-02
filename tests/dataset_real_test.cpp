#include <data/Dataset.hpp>
#include <hardware/DeviceManager.hpp>

#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace fs = std::filesystem;

static void require(bool condition, const char* message)
{
    if(!condition){
        throw std::runtime_error(message);
    }
}

int main()
{
    try
    {
        fs::path dataset_root = "datasets/perros";

        if(!fs::exists(dataset_root))
        {
            std::cout << "Dataset not found (" << dataset_root
                      << "). Skipping real dataset test.\n";
            return 0;
        }

        std::cout << "Loading dataset...\n";

        Dataset dataset(
            dataset_root,
            DeviceManager::get_cpu_device(),
            false,  // RGB
            true,   // normalize
            true    // recursive
        );

        std::cout << "\n========== DATASET SUMMARY ==========\n";

        std::cout << "Images : "
                  << dataset.size()
                  << '\n';

        std::cout << "Classes: "
                  << dataset.num_classes()
                  << '\n';

        std::cout << "\nClass list:\n";

        for(size_t i = 0; i < dataset.class_names().size(); ++i)
        {
            std::cout
                << "  ["
                << i
                << "] "
                << dataset.class_names()[i]
                << '\n';
        }

        size_t min_height = std::numeric_limits<size_t>::max();
        size_t min_width  = std::numeric_limits<size_t>::max();

        size_t max_height = 0;
        size_t max_width  = 0;

        for(size_t i = 0; i < dataset.size(); ++i)
        {
            const auto& sample = dataset[i];

            require(sample.image != nullptr,
                    "Null image pointer");

            require(sample.label < dataset.num_classes(),
                    "Invalid class label");

            require(fs::exists(sample.path),
                    "Image file does not exist");

            const Shape& shape = sample.image->shape;

            require(shape.n_dim == 4,
                    "Image tensor must be 4D");

            require(shape[0] == 1,
                    "Batch dimension must be 1");

            require(shape[1] == 3,
                    "Expected RGB images");

            size_t h = shape[2];
            size_t w = shape[3];

            min_height = std::min(min_height, h);
            min_width  = std::min(min_width, w);

            max_height = std::max(max_height, h);
            max_width  = std::max(max_width, w);
        }

        std::cout << "\nImage size statistics\n";

        std::cout
            << "Minimum : "
            << min_height
            << " x "
            << min_width
            << '\n';

        std::cout
            << "Maximum : "
            << max_height
            << " x "
            << max_width
            << '\n';

        std::cout
            << "\nVerifying shuffled indices...\n";

        auto shuffled = dataset.shuffled_indices(42);

        require(shuffled.size() == dataset.size(),
                "Invalid shuffle size");

        std::vector<bool> visited(dataset.size(), false);

        for(size_t idx : shuffled)
        {
            require(idx < dataset.size(),
                    "Shuffle index out of bounds");

            require(!visited[idx],
                    "Duplicate shuffle index");

            visited[idx] = true;
        }

        for(bool v : visited)
        {
            require(v,
                    "Missing shuffle index");
        }

        std::cout
            << "\nReal dataset test passed successfully.\n";

        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr
            << "\nReal dataset test failed:\n"
            << e.what()
            << '\n';

        return 1;
    }
}