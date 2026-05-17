#include <memory>
#include <vector>
#include <initializer_list>

enum class DeviceType{
    CPU,
    GPU
};

class Tensor{
private:
    DeviceType device = DeviceType::CPU;
    std::vector<int> shape = std::vector<int>(4,0);
    std::unique_ptr<std::vector<float>> data = nullptr;
    std::unique_ptr<float[]> gpu_data = nullptr;
public:
    Tensor(std::initializer_list<size_t>, float = 0.0f);
    Tensor(Tensor&);
    void to(DeviceType);
};