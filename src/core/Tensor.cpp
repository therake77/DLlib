#include <core/Tensor.hpp>
#include <stdexcept>
#include <cstring>

Tensor::Tensor(std::initializer_list<size_t> args, std::shared_ptr<Device> device)
: shape(args.begin(),args.end()), device(device), cpu_data(nullptr, [](void* ptr){}), gpu_data(nullptr, nullptr) {
    size_t total_size = 1;
    for(auto it = shape.begin(); it != shape.end(); it++){
        total_size*=*it;
    }

    //Data initialization
    void* data_ptr = device->allocate(total_size * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed to allocate memory for tensor");
    }
    this->device->memset(data_ptr, 1.0f, total_size * sizeof(float));

    auto curr_dev = this->device;
    this->cpu_data = std::unique_ptr<void,std::function<void(void*)>>(
        data_ptr,
        [curr_dev](void* ptr){
            curr_dev->free(ptr);
        }
    );
}

Tensor::Tensor(Tensor& t, std::shared_ptr<Device> device) : device(device) {
    size_t total_size = 1;
    for(auto it = t.shape.begin(); it != t.shape.end(); it++){
        total_size*=*it;
    }
    void* data_ptr = device->allocate(total_size * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }
    std::memcpy(data_ptr, t.cpu_data.get(), total_size * sizeof(float));
    this->cpu_data = std::unique_ptr<void,std::function<void(void*)>>(
        data_ptr,
        [device](void* ptr){
            device->free(ptr);
        }
    );
}

const float* Tensor::data() const{
    return static_cast<const float*>(this->cpu_data.get());
}

//TODO(Still tests are needed)
void Tensor::to(std::shared_ptr<Device> device){
    if(this->device->get_type() == device->get_type()) return;
    
    size_t total_data = 1;
    for(auto it = this->shape.begin(); it != this->shape.end(); it++){
        total_data*=*it;
    }
    void* data_ptr = device->allocate(total_data * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }
    
    if(device->get_type() == DeviceType::CPU){
        //Data is in GPU, copy to CPU
        device->copy_to_host(data_ptr, this->gpu_data.get(), total_data * sizeof(float));
        this->cpu_data = std::unique_ptr<void,std::function<void(void*)>>(
            data_ptr,
            [device](void* ptr){
                device->free(ptr);
            }
        );
    }else{
        //Data is in CPU, copy to GPU
        device->copy_to_device(data_ptr, this->cpu_data.get(), total_data * sizeof(float));
        this->gpu_data = std::unique_ptr<void,std::function<void(void*)>>(
            data_ptr,
            [device](void* ptr){
                device->free(ptr);
            }
        );
    }
    this->device = device;
}