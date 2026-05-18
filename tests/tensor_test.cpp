#include <core/Tensor.hpp>
#include <iostream>

int main(){
    auto device = DeviceManager::get_cpu_device();
    Tensor t = Tensor({2,3}, device);
    //Shape is (2,3), then total size is 6
    const float* data_ptr = t.data();
    for(size_t i = 0; i < 6; i++){
        std::cout << data_ptr[i] << " ";
        if(data_ptr[i] != 1.0f){
            std::cerr << "Test failed: Expected value 1.0f, got " << data_ptr[i] << std::endl;
            return 1;
        }
    }
    std::cout << "\nTest passed: All values are 1.0f as expected." << std::endl;
    return 0;
}