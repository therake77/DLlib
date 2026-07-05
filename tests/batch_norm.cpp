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

} // namespace

int main() {
    try {
        auto device = DeviceManager::get_cpu_device();

        float values[] = {1.0f, 2.0f, 3.0f, 4.0f};
        Tensor input({2, 1, 1, 2}, values, device);

        BatchNorm batch_norm(1, device);
        batch_norm.gamma.data()[0] = 1.0f;
        batch_norm.beta.data()[0] = 0.0f;
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

        std::cout << "BatchNorm tests passed" << std::endl;
        return 0;
    }
    catch(const std::exception& e) {
        std::cerr << "BatchNorm tests failed: " << e.what() << std::endl;
        return 1;
    }
}
