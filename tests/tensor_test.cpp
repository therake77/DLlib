#include <core/Tensor.hpp>
#include <kernel/TensorMath.hpp>
#include <iostream>

void printTensorAsRaw(Tensor& a){
    const float* data = a.data();
    for(size_t i = 0; i < a.size; i++){
        std::cout<<" "+std::to_string(data[i])+" "<<" ";
    }
    std::cout<<std::endl;
}

int main(){
    auto device = DeviceManager::get_cpu_device();
    Tensor a = Tensor({2,3}, device, 1.0f);
    Tensor b = Tensor({3,1},device, 3.0f);
    Tensor c = Tensor({2,3}, device, 5.0f);
    Tensor result1 = TensorMath::matmul(a,b);

    Tensor result2 = Tensor({2,3},device);
    BaseMath::add(a,c,result2);
    printTensorAsRaw(result2);
    BaseMath::sub(a,c,result2);
    printTensorAsRaw(result2);

    printTensorAsRaw(result1);

    std::cout << "\nTest passed" << std::endl;
    return 0;
}