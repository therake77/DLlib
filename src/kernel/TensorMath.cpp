#include <kernel/TensorMath.hpp>
#include <kernel/CpuMath.hpp>
#include <kernel/GpuMath.cuh>
#include <stdexcept>
#include <string>

Tensor TensorMath::add(Tensor& a, Tensor&b){
    //Device type
    DeviceType type = a.getDevice().get()->get_type();
    if(type != b.getDevice().get()->get_type()){ throw std::runtime_error("Tensor device type mismatch");}
    //Shape check
    if(a.size != b.size){ throw std::runtime_error("Shape mismatch"); }
    //Preallocate memory
    Tensor out = Tensor(a.getShape(),a.getDevice());
    if(type == DeviceType::CPU){
        CpuMath::add(a.data(),b.data(),a.size, out.data());
    }else{
        GpuMath::add(a.data(),b.data(),a.size,out.data());
    }
    return out;
}

Tensor TensorMath::sub(Tensor& a, Tensor&b){
    //Device type
    DeviceType type = a.getDevice().get()->get_type();
    if(type != b.getDevice().get()->get_type()){ throw std::runtime_error("Tensor device type mismatch");}
    //Shape check
    if(a.size != b.size){ throw std::runtime_error("Shape mismatch"); }
    //Preallocate memory
    Tensor out = Tensor(a.getShape(),a.getDevice());
    if(type == DeviceType::CPU){
        CpuMath::sub(a.data(),b.data(),a.size, out.data());
    }else{
        GpuMath::sub(a.data(),b.data(),a.size,out.data());
    }
    return out;
}

Tensor TensorMath::scale(Tensor& a, float _n){
    //Device type
    DeviceType type = a.getDevice().get()->get_type();
    //Preallocate memory
    Tensor out = Tensor(a.getShape(),a.getDevice());
    if(type == DeviceType::CPU){
        CpuMath::scale(a.data(), a.size, _n, out.data());
    }else{
        GpuMath::scale(a.data(), a.size, _n, out.data());
    }
    return out;
}

Tensor TensorMath::multiply(Tensor& a, Tensor& b){
    DeviceType type = a.getDevice().get()->get_type();
    if(type != b.getDevice().get()->get_type()){ throw std::runtime_error("Tensor device type mismatch");}
    //Shape check
    if(a.size != b.size){ throw std::runtime_error("Shape mismatch"); }
    //Preallocate memory
    Tensor out = Tensor(a.getShape(),a.getDevice());
    if(type == DeviceType::CPU){
        CpuMath::multiply(a.data(),b.data(),a.size, out.data());
    }else{
        GpuMath::multiply(a.data(),b.data(),a.size,out.data());
    }
    return out;
}


void BaseMath::add(Tensor& a, Tensor& b, Tensor& out){
    DeviceType type = a.getDevice().get()->get_type();
    //Device checks
    if(type != b.getDevice().get()->get_type() || type != out.getDevice().get()->get_type()){ throw std::runtime_error("Tensor device type mismatch");}
    //Shape checks
    if( (a.size ^ b.size ^ out.size) != 0 ){ throw std::runtime_error("Shape mismatch"); }

    if(type == DeviceType::CPU){
        CpuMath::add(a.data(),b.data(),a.size, out.data());
    }else{
        GpuMath::add(a.data(),b.data(),a.size,out.data());
    }
}
