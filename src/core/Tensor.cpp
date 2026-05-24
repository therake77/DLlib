#include <core/Tensor.hpp>
#include <stdexcept>
#include <cstring>

Tensor::Tensor(std::initializer_list<size_t> args, std::shared_ptr<Device> device)
: Tensor(std::vector<size_t>(args),device,0.0f){}

Tensor::Tensor(std::vector<size_t> shape, std::shared_ptr<Device> = DeviceManager::get_cpu_device(), float _fill = 0.0f)
: _shape(shape.begin(),shape.end()), device(device), cpu_data(nullptr, [](void* ptr){}), gpu_data(nullptr, nullptr){
    size_t total_size = 1;
    for(auto it = shape.begin(); it != shape.end(); it++){
        total_size*=*it;
    }
}

Tensor::Tensor(Tensor& t, std::shared_ptr<Device> device) : device(device), shape(t.shape), cpu_data(nullptr, [](void*){}), gpu_data(nullptr, [](void*){}), _size(t.size) {

    void* data_ptr = this->device->allocate(_size * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }

    auto src_type = t.getDevice().get()->get_type();
    auto dst_type = this->device->get_type();

    if(src_type == DeviceType::CPU && dst_type == DeviceType::CPU){
        std::memcpy(data_ptr, t.data(), _size * sizeof(float));
    }else if(src_type == DeviceType::CPU && dst_type == DeviceType::GPU){
        this->device->copy_to_device(data_ptr, t.data(), _size * sizeof(float));
    }else if(src_type == DeviceType::GPU && dst_type == DeviceType::CPU){
        this->device->copy_to_host(data_ptr, t.data(), _size * sizeof(float));
    }else{
        this->device->copy_device_to_device(data_ptr, t.data(), _size * sizeof(float));
    }

    auto curr_dev = this->device;
    auto deleter = [curr_dev](void* ptr){
        curr_dev->free(ptr);
    };

    if(this->device->get_type() == DeviceType::CPU){
        this->cpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->gpu_data.reset();
    }else{
        this->gpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->cpu_data.reset();
    }
}

float* Tensor::data(){
    if( this->device.get()->get_type() == DeviceType::CPU){
        return static_cast<float*>(this->cpu_data.get());
    }else{
        return static_cast<float*>(this->gpu_data.get());
    }
}

const float* Tensor::data() const{
    if( this->device.get()->get_type() == DeviceType::CPU){
        return static_cast<float*>(this->cpu_data.get());
    }else{
        return static_cast<float*>(this->gpu_data.get());
    }
}

//TODO(Still tests are needed)
void Tensor::to(std::shared_ptr<Device> device){
    if(this->device->get_type() == device->get_type()) return;
    
    void* data_ptr = device->allocate(_size * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }
    
    auto curr_dev = device;
    auto deleter = [curr_dev](void* ptr){
        curr_dev->free(ptr);
    };

    if(device->get_type() == DeviceType::CPU){
        //Data is in GPU, copy to CPU
        device->copy_to_host(data_ptr, this->gpu_data.get(), _size * sizeof(float));
        this->cpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->gpu_data.reset();
    }else{
        //Data is in CPU, copy to GPU
        device->copy_to_device(data_ptr, this->cpu_data.get(), _size * sizeof(float));
        this->gpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->cpu_data.reset();
    }
    this->device = device;
}
