#include <iostream>
#include <core/ConvLayer.hpp>
#include <core/Pooling.hpp>
#include <core/ActivationLayer.hpp>
#include <core/BatchNormLayer.hpp>
#include <core/Flatten.hpp>
#include <core/OutputLayer.hpp>
#include <core/FullyConnected.hpp>

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
}

int main(){
    auto device = DeviceManager::get_cpu_device();
    Conv2D conv_layer = Conv2D(
        {2,1,3,3},
        device,
        {0,0},
        {1,1}
    );
    BatchNorm batch_layer = BatchNorm(
        2,device
    );
    MaxPooling max_pool_layer = MaxPooling(
        {2,2},
        {2,2},
        {0,0},
        device
    );
    
    Flatten flatten_layer = Flatten(
        device
    );
    
    ActivationLayer activation = ActivationLayer(
        ActivationTypes::ReLU,
        device
    );
    
    FullyConnected dense1 = FullyConnected(
        {8,4},
        device
    );
    
    ActivationLayer activation2 = ActivationLayer(
        ActivationTypes::ReLU,
        device
    );

    FullyConnected dense2 = FullyConnected(
        {4,2},
        device
    );

    OutputLayer out = OutputLayer(
        device,
        2
    );

    //Full pipeline
    Tensor input = Tensor(
        {1,1,6,6},
        device,
        1.0f,
        true
    );
    std::cout<<"Input:\n";
    fancy_print(input);
    //Conv2D
    std::cout<<"Con2D:\n";
    auto result1 = conv_layer.forward(input);
    fancy_print(*result1);
    std::cout<<"BatchNorm:\n";
    auto result2 = batch_layer.forward(*result1);
    fancy_print(*result2);
    std::cout<<"MaxPool:\n";
    auto result3 = max_pool_layer.forward(*result2);
    fancy_print(*result3);
    std::cout<<"Flatten:\n";
    auto result4 = flatten_layer.forward(*result3);
    fancy_print(*result4);
    std::cout<<"Activation1:\n";
    auto result5 = activation.forward(*result4);
    fancy_print(*result5);
    std::cout<<"FullyConnected1:\n";
    auto result6 = dense1.forward(*result5);
    fancy_print(*result6);
    std::cout<<"Activation2:\n";
    auto result7 = activation2.forward(*result6);
    fancy_print(*result7);
    std::cout<<"FullyConnected2:\n";
    auto result8 = dense2.forward(*result7);
    fancy_print(*result8);
    std::cout<<"Sotfmax+CrossEntropy:\n";
    auto output = out.forward(*result8);
    std::cout<<"Output\n";
    fancy_print(
        *output
    );

    //Now it comes the backward pass
    //Let the label corresponding to the current batch:
    Tensor label = Tensor(
        {1,2},
        device,
        1.0f,
        true
    );
    
    //Feed it to the last layer
    auto backward_grad1 = out.backward(label,*result8);
    std::cout<<"backward_grad1 passed\n";
    auto backward_grad2 = dense2.backward(*backward_grad1,*result7);
    std::cout<<"backward_grad2 passed\n";
    auto backward_grad3 = activation2.backward(*backward_grad2,*result6);
    std::cout<<"backward_grad3 passed\n";
    auto backward_grad4 = dense1.backward(*backward_grad3,*result5);
    std::cout<<"backward_grad4 passed\n";
    auto backward_grad5 = activation.backward(*backward_grad4,*result4);
    std::cout<<"backward_grad5 passed\n";
    auto backward_grad6 = flatten_layer.backward(*backward_grad5,*result3);
    std::cout<<"backward_grad6 passed\n";
    auto backward_grad7 = max_pool_layer.backward(*backward_grad6,*result2);
    std::cout<<"backward_grad7 passed\n";
    auto backward_grad8 = batch_layer.backward(*backward_grad7,*result1);
    std::cout<<"backward_grad8 passed\n";
    auto backward_grad9 = conv_layer.backward(*backward_grad8,input);
    fancy_print(
        *backward_grad9
    );
}