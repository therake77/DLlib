#include <core/Layer.hpp>
#include <functional>

enum class ActivationTypes{
    ReLU,
    Sigmoid
};

class ActivationLayer : public Layer{
private:
    ActivationTypes type;
    Tensor _out;
public:

    ActivationLayer(
        std::array<std::size_t, TENSOR_MAX_DIM>,
        ActivationTypes = ActivationTypes::ReLU,
        std::shared_ptr<Device> = DeviceManager::get_cpu_device()
    );

    Tensor& forward(Tensor&) override;
    void update_weights(Tensor&) override;

};
