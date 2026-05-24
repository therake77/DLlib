#include <core/Tensor.hpp>
#include <stdexcept>
#include <cstring>

namespace {
    size_t compute_total_size(const std::vector<size_t>& shape) {
        size_t total_size = 1;
        for (size_t dim : shape) {
            total_size *= dim;
        }
        return total_size;
    }
}

Tensor::Tensor(std::initializer_list<size_t> args, std::shared_ptr<Device> device)
: shape(args.begin(),args.end()), device(device), cpu_data(nullptr, [](void*){}), gpu_data(nullptr, [](void*){}), n_elem(compute_total_size(shape)) {
    
    //Data initialization
    void* data_ptr = this->device->allocate(n_elem * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed to allocate memory for tensor");
    }
    this->device->memset(data_ptr, 1.0f, n_elem * sizeof(float));

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

Tensor::Tensor(const std::vector<size_t>& args, std::shared_ptr<Device> device)
: shape(args), device(device), cpu_data(nullptr, [](void*){}), gpu_data(nullptr, [](void*){}), n_elem(compute_total_size(shape)) {

    void* data_ptr = this->device->allocate(n_elem * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed to allocate memory for tensor");
    }
    this->device->memset(data_ptr, 1.0f, n_elem * sizeof(float));

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

Tensor::Tensor(const Tensor& t, std::shared_ptr<Device> device) : device(device), shape(t.shape), cpu_data(nullptr, [](void*){}), gpu_data(nullptr, [](void*){}), n_elem(t.n_elem) {

    void* data_ptr = this->device->allocate(n_elem * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }

    auto src_type = t.get_device()->get_type();
    auto dst_type = this->device->get_type();

    if(src_type == DeviceType::CPU && dst_type == DeviceType::CPU){
        std::memcpy(data_ptr, t.data(), n_elem * sizeof(float));
    }else if(src_type == DeviceType::CPU && dst_type == DeviceType::GPU){
        this->device->copy_to_device(data_ptr, t.data(), n_elem * sizeof(float));
    }else if(src_type == DeviceType::GPU && dst_type == DeviceType::CPU){
        this->device->copy_to_host(data_ptr, t.data(), n_elem * sizeof(float));
    }else{
        this->device->copy_device_to_device(data_ptr, t.data(), n_elem * sizeof(float));
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

const float* Tensor::data() const{
    if(this->device->get_type() == DeviceType::CPU){
        return static_cast<const float*>(this->cpu_data.get());
    }
    return static_cast<const float*>(this->gpu_data.get());
}

float* Tensor::data(){
    if(this->device->get_type() == DeviceType::CPU){
        return static_cast<float*>(this->cpu_data.get());
    }
    return static_cast<float*>(this->gpu_data.get());
}

const std::vector<size_t>& Tensor::get_shape() const{
    return this->shape;
}

size_t Tensor::size() const{
    return this->n_elem;
}

std::shared_ptr<Device> Tensor::get_device() const{
    return this->device;
}

//TODO(Still tests are needed)
void Tensor::to(std::shared_ptr<Device> device){
    if(this->device->get_type() == device->get_type()) return;
    
    void* data_ptr = device->allocate(n_elem * sizeof(float));
    if(data_ptr == nullptr){
        throw std::runtime_error("Failed memory allocation");
    }
    
    auto curr_dev = device;
    auto deleter = [curr_dev](void* ptr){
        curr_dev->free(ptr);
    };

    if(device->get_type() == DeviceType::CPU){
        //Data is in GPU, copy to CPU
        device->copy_to_host(data_ptr, this->gpu_data.get(), n_elem * sizeof(float));
        this->cpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->gpu_data.reset();
    }else{
        //Data is in CPU, copy to GPU
        device->copy_to_device(data_ptr, this->cpu_data.get(), n_elem * sizeof(float));
        this->gpu_data = std::unique_ptr<void,std::function<void(void*)>>(data_ptr, deleter);
        this->cpu_data.reset();
    }
    this->device = device;
}