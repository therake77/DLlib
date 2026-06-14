#include <iostream>
#include <core/Tensor.hpp>
#include <hardware/DeviceManager.hpp>
#include <math/TensorMath.hpp>

void fancy_print(Tensor& a){
    const float* data = a.data();
    std::vector<size_t> coords(a.shape.n_dim,0);
    for(size_t i = 0; i < a.shape.n_dim; ++i){
        std::cout<<"[";
    }
    for(size_t el = 0; el < a.shape.n_elements; ++el){
        size_t flat_idx = 0;
        for (size_t i = 0; i < a.shape.n_dim; ++i){
            flat_idx+=coords[i] * a.shape._strides[i];
        }
        std::cout << data[flat_idx];
        bool finished = false;
        size_t reset = 0;
        for(size_t i = a.shape.n_dim; i --> 0; ){
            coords[i]++;
            if(coords[i] < a.shape[i]){ break; }
            else{ if( i ==0 ){ finished = true; }else{ coords[i] = 0; reset++; } }
        }

        if(finished || el == a.shape.n_elements -1 ){
            for(size_t i = 0; i < a.shape.n_dim; ++i) std::cout<<"]";
            std::cout<< std::endl;
            break;
        }

        if(reset == 0){std::cout<<", ";}
        else{
            for(size_t i = 0; i < reset; ++i){std::cout<<"]";}
            std::cout<<",\n";
            for(size_t i = 0; i < a.shape.n_dim - reset; ++i){ std::cout<< " "; }
            for(size_t i = 0; i < reset; ++i){ std::cout<<"["; } 
        }
    }
}

void log(std::string s){
    std::cout<<s<<std::endl;
}

int main(){
    std::cout<<" Conv2D Test "<<std::endl;
    
    auto device_shared_ptr = DeviceManager::get_cpu_device(); 

    Tensor input = Tensor(
        {1,1,4,4},device_shared_ptr,2.0f
    );
    Tensor kernel = Tensor(
        {1,1,3,3},device_shared_ptr,1.0f
    );
    Tensor output = Tensor(
        {1,1,4,4},device_shared_ptr,0.0f
    );
    TensorMath::conv2D(input,kernel,output,1,1);
    log("Input:");
    fancy_print(input);
    log("Kernel:");
    fancy_print(kernel);
    log("Output:");
    fancy_print(output);
    return 0;
}