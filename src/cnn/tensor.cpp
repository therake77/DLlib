#include <cnn/tensor.hpp>
#include <cnn/backends/gpu_backend.cu>
#include <stdexcept>

Tensor::Tensor(std::initializer_list<size_t> args, float default_value = 0.0f){
    size_t total_size = 1;
    
    //Shape parsing
    for(auto t = args.begin(); t != args.end(); t+=1 ){
        if(*t < 0){
            throw std::invalid_argument("Shape must be no negative");
        }else{
            total_size*=*t;
            (this->shape).push_back(*t);
        }
    }

    //Data initialization
    (this->data)->reserve(total_size);

    size_t index = 0;
    while(index < total_size){
        (*(this->data))[index] = default_value;
        index++;
    }
}

Tensor::Tensor(Tensor& t){
    //Just copy the elements
    size_t total_size = 1;
    for(size_t t : t.shape ){
        total_size*=t;
    }
    size_t index = 0;
    (this->data)->reserve(total_size);
    while(index < total_size){
        (*(this->data))[index] = (*(t.data))[index];
    }
}

//TODO(Finish this)
void Tensor::to(DeviceType dev){
    if(this->device == dev) return;
    switch(dev){
        case DeviceType::CPU:{
            //GPU -> CPU
            float* data = GPU::getFromMemory((this->gpu_data).get());
            
            GPU::freeMemory(this->gpu_data.get());
            this->gpu_data.reset(nullptr);
        }
        case DeviceType::GPU:{
            //GPU -> CPU
            //Allocate memory for the tensor in the gpu
            float* data_ptr = GPU::allocateMemory(*(this->data));
            this->gpu_data.reset(data_ptr);
        }
        default : throw std::invalid_argument("Unknown device type");
    }
}