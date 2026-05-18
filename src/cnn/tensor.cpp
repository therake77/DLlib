#include <cnn/tensor.hpp>
#include <stdexcept>
#include <cnn/backends/gpu_backend.cuh>

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
    this->cpu_data = std::vector<float>(total_size,default_value);
}

Tensor::Tensor(Tensor& t) : cpu_data(t.cpu_data) {}

//TODO(Finish this)
void Tensor::to(DeviceType dev){
    if(this->device == dev) return;
    size_t total_data = 1;
    for(auto it = this->shape.begin(); it != this->shape.end(); it++){
        total_data*=*it;
    }

    switch(dev){
        case DeviceType::CPU:{
            //GPU -> CPU
            (this->cpu_data).assign(
                *(this->gpu_data.ptr),
                this->gpu_data.size
            );
            GPU::free_memory(this->gpu_data);
        }
        
        case DeviceType::GPU:{
            //CPU -> GPU
            //Allocate memory for the tensor in the gpu
            //Do not touch cpu mem
            this->gpu_data= std::move(
                GPU::allocate_memory((this->cpu_data).data(), this->cpu_data.size())
            );
        }

        default : throw std::invalid_argument("Unknown device type");
    }
}