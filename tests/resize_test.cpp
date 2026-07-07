#include <data/transforms/Compose.hpp>
#include <data/transforms/Resize.hpp>
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
        throw std::runtime_error("Unexpected resized tensor value");
    }
}

int main() {
    try {
        Tensor input(
            {1, 1, 2, 2},
            DeviceManager::get_cpu_device(),
            0.5f,
            true
        );

        Resize resize(4, 4);
        Tensor output = resize(input);

        require(
            output.shape.n_dim == 4,
            "Resized tensor should be 4D"
        );

        require(
            output.shape[0] == 1,
            "Resized batch dimension should be 1"
        );

        require(
            output.shape[1] == 1,
            "Resized channel dimension should be 1"
        );

        require(
            output.shape[2] == 4,
            "Resized height should be 4"
        );

        require(
            output.shape[3] == 4,
            "Resized width should be 4"
        );

        const float* data = output.data();

        for(std::size_t i = 0; i < output.shape.n_elements; ++i) {
            check_close(data[i], 0.5f);
        }

        Compose compose({
            std::make_shared<Resize>(3, 5)
        });

        Tensor composed_output = compose(input);

        require(
            composed_output.shape[0] == 1,
            "Composed resized batch dimension should be 1"
        );

        require(
            composed_output.shape[1] == 1,
            "Composed resized channel dimension should be 1"
        );

        require(
            composed_output.shape[2] == 3,
            "Composed resized height should be 3"
        );

        require(
            composed_output.shape[3] == 5,
            "Composed resized width should be 5"
        );

        std::cout
            << "Resize test passed"
            << std::endl;

        return 0;
    }
    catch(const std::exception& e) {
        std::cerr
            << "Resize test failed: "
            << e.what()
            << std::endl;

        return 1;
    }
}
