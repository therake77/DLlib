#include <core/BatchNormLayer.hpp>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require_close(float actual, float expected, const char* message) {
    if(std::fabs(actual - expected) > 1e-4f) {
        throw std::runtime_error(message);
    }
}

void require_close_tol(float actual, float expected, float tolerance, const char* message) {
    if(std::fabs(actual - expected) > tolerance) {
        throw std::runtime_error(message);
    }
}

} // namespace

int main() {
    try {
        auto device = DeviceManager::get_cpu_device();

        float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
        Tensor input({2, 1, 1, 2}, values, device);

        BatchNorm batch_norm(1, device);
        require_close(batch_norm.gamma.data()[0], 1.0f, "BatchNorm gamma should initialize to one");
        require_close(batch_norm.beta.data()[0], 0.0f, "BatchNorm beta should initialize to zero");
        batch_norm.set_mode(LayerMode::Training);

        auto output = batch_norm.forward(input);
        const float* out = output->data();

        float output_sum = 0.0f;
        for(std::size_t i = 0; i < output->shape.n_elements; ++i) {
            output_sum += out[i];
        }
        require_close(output_sum, 0.0f, "BatchNorm normalized output should sum to zero per channel");

        Tensor grad(output->shape._shapes, device, 1.0f, true);
        auto backward = batch_norm.backward(grad, input);
        const float* dx = backward->data();

        for(std::size_t i = 0; i < backward->shape.n_elements; ++i) {
            require_close(dx[i], 0.0f, "BatchNorm backward should produce zero dx for all-one upstream grad");
        }

        float grad_values[] = {0.4f, -0.3f, 0.2f, -0.1f};
        Tensor upstream({2, 1, 1, 2}, grad_values, device);
        batch_norm.forward(input);
        backward = batch_norm.backward(upstream, input);
        dx = backward->data();

        constexpr float eps = 1e-3f;
        for(std::size_t i = 0; i < 4; ++i) {
            float plus_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
            float minus_values[] = {1.0f, 2.0f, 3.0f, 4.0f};
            plus_values[i] += eps;
            minus_values[i] -= eps;

            Tensor plus_input({2, 1, 1, 2}, plus_values, device);
            Tensor minus_input({2, 1, 1, 2}, minus_values, device);

            auto plus_output = batch_norm.forward(plus_input);
            float plus_loss = 0.0f;
            for(std::size_t j = 0; j < plus_output->shape.n_elements; ++j) {
                plus_loss += plus_output->data()[j] * grad_values[j];
            }

            auto minus_output = batch_norm.forward(minus_input);
            float minus_loss = 0.0f;
            for(std::size_t j = 0; j < minus_output->shape.n_elements; ++j) {
                minus_loss += minus_output->data()[j] * grad_values[j];
            }

            float numerical = (plus_loss - minus_loss) / (2.0f * eps);
            require_close_tol(dx[i], numerical, 2e-3f, "BatchNorm dx should match finite differences");
        }

        std::cout << "BatchNorm tests passed" << std::endl;
        return 0;
    }
    catch(const std::exception& e) {
        std::cerr << "BatchNorm tests failed: " << e.what() << std::endl;
        return 1;
    }
}
