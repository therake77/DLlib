#include <memory>
#include <vector>
#include <initializer_list>
#include <types/memory.cuh>

enum class DeviceType{
    CPU,
    GPU
};

class Tensor{
private:
    DeviceType device = DeviceType::CPU;
    std::vector<size_t> shape = std::vector<size_t>(4,0);
    std::vector<float> cpu_data;
    CudaMemPtr gpu_data;
public:
    Tensor(std::initializer_list<size_t>, float = 0.0f);
    Tensor(Tensor&);
    void to(DeviceType);
};