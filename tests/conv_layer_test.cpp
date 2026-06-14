#include <hardware/DeviceManager.hpp>
#include <core/ConvLayer.hpp>
#include <iostream>

inline void fancy_print(const Tensor& a){
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
    std::cout<<std::endl;
}

int main(){
    auto device = DeviceManager::get_cpu_device();
    Tensor input = Tensor(
        {1,3,10,10},device
    );
    fancy_print(input);
    Conv2D layer = Conv2D( {1,3,3,3},device,{0,0},{1,1} );
    fancy_print(layer.kernels);
    Tensor output = layer.forward(input);
    fancy_print(output);

}