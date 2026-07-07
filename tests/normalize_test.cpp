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
        throw std::runtime_error("Unexpected normalized tensor value");
    }
}

int main() {
    try {
        Tensor input(
            {1, 2, 1, 2},
            DeviceManager::get_cpu_device(),
            0.0f,
            true
        );

        float* data = input.data();
        data[0] = 1.0f;
        data[1] = 3.0f;
        data[2] = 10.0f;
        data[3] = 14.0f;

        Normalize normalize(
            {1.0f, 10.0f},
            {2.0f, 4.0f}
        );

        Tensor output = normalize(input);

        require(
            output.shape == input.shape,
            "Normalize should preserve input shape"
        );

        const float* normalized = output.data();

        check_close(normalized[0], 0.0f);
        check_close(normalized[1], 1.0f);
        check_close(normalized[2], 0.0f);
        check_close(normalized[3], 1.0f);

        Compose compose({
            std::make_shared<Normalize>(1.0f, 2.0f)
        });

        Tensor composed = compose(input);
        const float* composed_data = composed.data();

        check_close(composed_data[0], 0.0f);
        check_close(composed_data[1], 1.0f);
        check_close(composed_data[2], 4.5f);
        check_close(composed_data[3], 6.5f);

        std::cout
            << "Normalize test passed"
            << std::endl;

        return 0;
    }
    catch(const std::exception& e) {
        std::cerr
            << "Normalize test failed: "
            << e.what()
            << std::endl;

        return 1;
    }
}
