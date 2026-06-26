
#ifndef LLCNN_TENSOR
    #define LLCNN_TENSOR

    #include <memory>
    #include <vector>
    #include <initializer_list>
    #include <hardware/Device.hpp>
    #include <hardware/DeviceManager.hpp>
    #include <functional>
    #include <array>
    #include <core/Shape.hpp>
    /*
    Clase contenedora de datos. Representa un tensor de como máximo TENSOR_MAX_DIM dimensiones (x,y,z,w)
    */
    class Tensor{
    private:
        std::shared_ptr<Device> device;
        std::unique_ptr<void, std::function<void(void*)>> cpu_data;
        std::unique_ptr<void, std::function<void(void*)>> gpu_data;

    public:
        Shape shape;
        Tensor(std::array<size_t,TENSOR_MAX_DIM>, std::shared_ptr<Device> = DeviceManager::get_cpu_device(), float = 0.0f, bool = false);    
        Tensor(Tensor&, std::shared_ptr<Device> = DeviceManager::get_cpu_device());
        float* data();
        const float* data() const;
        void to(std::shared_ptr<Device>);
        const inline std::shared_ptr<Device>& getDevice() const { return this->device; }
        void resize(Shape s);

        Tensor& unsqueeze(std::size_t axis);
        void inner_expand(std::size_t dims);
        void outer_expand(std::size_t dims);
    };

#endif