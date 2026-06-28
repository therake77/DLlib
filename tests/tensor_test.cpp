#include <core/Tensor.hpp>
#include <hardware/DeviceManager.hpp>
#include <math/TensorMath.hpp>
#include <iostream>

void printTensorAsRaw(Tensor& a){
    const float* data = a.data();
    for(size_t i = 0; i < a.shape.n_elements; i++){
        std::cout<<" "+std::to_string(data[i])+" "<<" ";
    }
    std::cout<<std::endl;
}

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

void matmul_test(std::shared_ptr<Device> device){
    Tensor a = Tensor({2,1,1}, device, 7.0f,true);
    Tensor b = Tensor({1,3}, device, 2.0f,true);
    Tensor c = Tensor({2,1,3}, device,0.0f,true);
    try{
        fancy_print(a);
        fancy_print(b);
        TensorMath::matmul(a,b,c);
    }catch(std::exception){
        std::cout<<"matmul_test: Test failed";
    }
    fancy_print(c);
    std::cout<<"matmul_test: Test successful";
}

int main(){
    auto device = DeviceManager::get_cpu_device();
    matmul_test(device);

    std::cout << "\nTest passed" << std::endl;
    return 0;
}