#include <data/transforms/CenterCrop.hpp>
#include <data/transforms/Compose.hpp>
#include <data/transforms/Normalize.hpp>
#include <hardware/DeviceManager.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

static void require(bool condition, const char* message) {
    if(!condition) {
        throw std::runtime_error(message);
    }
}

static void check_close(float value, float expected, float eps = 1e-5f) {
    if(std::fabs(value - expected) > eps) {
        throw std::runtime_error("Unexpected center crop tensor value");
    }
}

int main() {
    try {
        Tensor input(
            {1, 1, 4, 4},
            DeviceManager::get_cpu_device(),
            0.0f,
            true
        );

        float* input_data = input.data();
        for(std::size_t i = 0; i < input.shape.n_elements; ++i) {
            input_data[i] = static_cast<float>(i);
        }

        CenterCrop crop(2, 2);
        Tensor output = crop(input);

        require(
            output.shape.n_dim == 4,
            "CenterCrop output should be 4D"
        );

        require(
            output.shape[0] == 1,
            "CenterCrop batch dimension should be 1"
        );

        require(
            output.shape[1] == 1,
            "CenterCrop channel dimension should be 1"
        );

        require(
            output.shape[2] == 2,
            "CenterCrop height should be 2"
        );

        require(
            output.shape[3] == 2,
            "CenterCrop width should be 2"
        );

        const float* output_data = output.data();

        check_close(output_data[0], 5.0f);
        check_close(output_data[1], 6.0f);
        check_close(output_data[2], 9.0f);
        check_close(output_data[3], 10.0f);

        Compose compose({
            std::make_shared<CenterCrop>(2, 2),
            std::make_shared<Normalize>(5.0f, 5.0f)
        });

        Tensor composed = compose(input);
        const float* composed_data = composed.data();

        check_close(composed_data[0], 0.0f);
        check_close(composed_data[1], 0.2f);
        check_close(composed_data[2], 0.8f);
        check_close(composed_data[3], 1.0f);

        std::cout
            << "CenterCrop test passed"
            << std::endl;

        return 0;
    }
    catch(const std::exception& e) {
        std::cerr
            << "CenterCrop test failed: "
            << e.what()
            << std::endl;

        return 1;
    }
}
