#include <core/Pooling.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if(!condition) {
        throw std::runtime_error(message);
    }
}

void require_close(float actual, float expected, const char* message) {
    if(std::fabs(actual - expected) > 1e-5f) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        auto device = DeviceManager::get_cpu_device();

        float increasing_values[25];
        for(std::size_t i = 0; i < 25; ++i) {
            increasing_values[i] = static_cast<float>(i + 1);
        }

        Tensor max_input({1, 1, 5, 5}, increasing_values, device);
        MaxPooling max_pool({2, 2}, {2, 2}, {0, 0}, device);

        auto max_output = max_pool.forward(max_input);
        require(max_output->shape == Shape({1, 1, 2, 2}), "MaxPool forward shape mismatch");

        const float* max_out = max_output->data();
        require_close(max_out[0], 7.0f, "Unexpected max pool value at 0");
        require_close(max_out[1], 9.0f, "Unexpected max pool value at 1");
        require_close(max_out[2], 17.0f, "Unexpected max pool value at 2");
        require_close(max_out[3], 19.0f, "Unexpected max pool value at 3");

        Tensor max_grad({1, 1, 2, 2}, device, 1.0f, true);
        auto max_backward = max_pool.backward(max_grad, max_input);
        const float* max_back = max_backward->data();

        for(std::size_t i = 0; i < 25; ++i) {
            float expected =
                (i == 6 || i == 8 || i == 16 || i == 18) ? 1.0f : 0.0f;
            require_close(max_back[i], expected, "Unexpected max pool backward value");
        }

        Tensor avg_input({1, 1, 4, 4}, device, 1.0f, true);
        AvgPooling avg_pool({2, 2}, {2, 2}, {0, 0}, device);

        auto avg_output = avg_pool.forward(avg_input);
        require(avg_output->shape == Shape({1, 1, 2, 2}), "AvgPool forward shape mismatch");

        const float* avg_out = avg_output->data();
        for(std::size_t i = 0; i < avg_output->shape.n_elements; ++i) {
            require_close(avg_out[i], 1.0f, "Unexpected avg pool forward value");
        }

        Tensor avg_grad({1, 1, 2, 2}, device, 1.0f, true);
        auto avg_backward = avg_pool.backward(avg_grad, avg_input);
        const float* avg_back = avg_backward->data();

        for(std::size_t i = 0; i < avg_backward->shape.n_elements; ++i) {
            require_close(avg_back[i], 0.25f, "Unexpected avg pool backward value");
        }

        std::cout << "Pooling tests passed" << std::endl;
        return 0;
    }
    catch(const std::exception& e) {
        std::cerr << "Pooling tests failed: " << e.what() << std::endl;
        return 1;
    }
}
