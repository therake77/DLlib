#include <kernel/TensorMath.hpp>
#include <kernel/CpuMath.hpp>
#include <kernel/GpuMath.cuh>
#include <stdexcept>
#include <string>

namespace {
	void ensure_same_shape(const Tensor& a, const Tensor& b) {
		if (a.size != b.size || a.shape != b.shape) {
			throw std::runtime_error("Tensor shape mismatch");
		}
	}
    void ensure_same_device(const Tensor& a, const Tensor& b){
        if (a.getDevice().get()->get_type() != b.getDevice()->get_type()){
            throw std::runtime_error("Tensor device mismatch");
        }
    }
}

//Element-wise operations
Tensor TensorMath::add(Tensor& a, Tensor&b){
    ensure_same_device(a,b);
    ensure_same_shape(a,b);
    //Device type
    DeviceType type = a.getDevice().get()->get_type();
    //Preallocate memory
    Tensor out = Tensor(a.shape,a.getDevice());
    BaseMath::add(a,b,out);
    return out;
}

Tensor TensorMath::sub(Tensor& a, Tensor&b){
    ensure_same_device(a,b);
    ensure_same_shape(a,b);
    //Device type
    DeviceType type = a.getDevice().get()->get_type();
    //Preallocate memory
    Tensor out = Tensor(a.shape,a.getDevice());
    BaseMath::sub(a,b,out);
    return out;
}

Tensor TensorMath::scale(Tensor& a, float _n){
    //Device type
    DeviceType type = a.getDevice().get()->get_type();
    //Preallocate memory
    Tensor out = Tensor(a.shape,a.getDevice());
    BaseMath::scale(a,_n,out);
    return out;
}

Tensor TensorMath::multiply(Tensor& a, Tensor& b){
    ensure_same_device(a,b);
    ensure_same_shape(a,b);
    //Device type
    DeviceType type = a.getDevice().get()->get_type();
    //Preallocate memory
    Tensor out = Tensor(a.shape,a.getDevice());
    BaseMath::multiply(a,b,out);
    return out;
}

//Activation functions
Tensor TensorMath::reLU(Tensor& a) {
    Tensor out(a.shape, a.getDevice());
    BaseMath::reLU(a, out);
    return out;
}

Tensor TensorMath::sigmoid(Tensor& a) {
    Tensor out(std::vector<size_t>(a.shape), a.getDevice());
    BaseMath::sigmoid(a, out);
    return out;
}

Tensor TensorMath::tanh(Tensor& a) {
    Tensor out(std::vector<size_t>(a.shape), a.getDevice());
    BaseMath::tanh(a, out);
    return out;
}

Tensor TensorMath::softmax(Tensor& a) {
    Tensor out(std::vector<size_t>(a.shape), a.getDevice());
    BaseMath::softmax(a, out);
    return out;
}

//Linear Algebra
Tensor TensorMath::matmul(Tensor& a, Tensor& b) {
    const auto& a_shape = a.shape;
    const auto& b_shape = b.shape;
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("matmul requires 2D tensors");
    }
    if (a_shape[1] != b_shape[0]) {
        throw std::runtime_error("matmul shape mismatch");
    }
    std::vector<size_t> out_shape = { a_shape[0], b_shape[1] };
    Tensor out(out_shape, a.getDevice());
    BaseMath::matmul(a, b, out);
    return out;
}

Tensor TensorMath::conv(Tensor& in, Tensor& ker, size_t stride, size_t padding) {
    const auto& in_shape = in.shape;
    const auto& ker_shape = ker.shape;
    if (in_shape.size() != 2 || ker_shape.size() != 2) {
        throw std::runtime_error("conv requires 2D tensors");
    }
    size_t in_h = in_shape[0];
    size_t in_w = in_shape[1];
    size_t k_h = ker_shape[0];
    size_t k_w = ker_shape[1];
    size_t padded_h = in_h + 2 * padding;
    size_t padded_w = in_w + 2 * padding;
    if (padded_h < k_h || padded_w < k_w) {
        throw std::runtime_error("conv kernel larger than input");
    }
    if (((padded_h - k_h) % stride) != 0 || ((padded_w - k_w) % stride) != 0) {
        throw std::runtime_error("conv stride/padding mismatch");
    }
    size_t out_h = (padded_h - k_h) / stride + 1;
    size_t out_w = (padded_w - k_w) / stride + 1;
    std::vector<size_t> out_shape = { out_h, out_w };
    
    Tensor out(out_shape, in.getDevice());
    BaseMath::conv(in, ker, out, stride, padding);
    return out;
}


//BaseMath operations

void BaseMath::add(Tensor& a, Tensor& b, Tensor& out){
    //No validations: Those were supposed to occurr at top TensorMath level API
    //TODO(Is really omit validations a good idea?)
    DeviceType type = a.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::add(a.data(),b.data(),a.size, out.data());
    }else{
        GpuMath::add(a.data(),b.data(),a.size,out.data());
    }

}

void BaseMath::sub(Tensor& a, Tensor& b, Tensor& out) {
    DeviceType type = a.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::sub(a.data(),b.data(),a.size, out.data());
    }else{
        GpuMath::sub(a.data(),b.data(),a.size,out.data());
    }
}

void BaseMath::scale(Tensor& a, float n, Tensor& out) {
    DeviceType type = a.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::scale(a.data(),a.size,n, out.data());
    }else{
        GpuMath::scale(a.data(),a.size,n, out.data());
    }
}

void BaseMath::multiply(Tensor& a, Tensor& b, Tensor& out) {
    DeviceType type = a.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::multiply(a.data(),b.data(),a.size, out.data());
    }else{
        GpuMath::multiply(a.data(),b.data(),a.size, out.data());
    }
}

//Activation functions
void BaseMath::reLU(Tensor& a, Tensor& out) {
    DeviceType type = a.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::reLU(a.data(),a.size, out.data());
    }else{
        GpuMath::reLU(a.data(),a.size, out.data());
    }
}

void BaseMath::sigmoid(Tensor& a, Tensor& out) {
    DeviceType type = a.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::sigmoid(a.data(),a.size, out.data());
    }else{
        GpuMath::sigmoid(a.data(),a.size, out.data());
    }
}

void BaseMath::tanh(Tensor& a, Tensor& out) {
    DeviceType type = a.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::tanh(a.data(),a.size, out.data());
    }else{
        GpuMath::tanh(a.data(),a.size, out.data());
    }
}

void BaseMath::softmax(Tensor& a, Tensor& out) {
    DeviceType type = a.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::softmax(a.data(),a.size, out.data());
    }else{
        GpuMath::softmax(a.data(),a.size, out.data());
    }
}

//Linear Algebra
void BaseMath::matmul(Tensor& a, Tensor& b, Tensor& out) {
    DeviceType type = a.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::matmul(a.data(),a.shape,b.data(), b.shape, out.data());
    }else{
        GpuMath::matmul(a.data(),a.shape,b.data(), b.shape, out.data());
    }
}

void BaseMath::conv(Tensor& in, Tensor& ker, Tensor& out, size_t stride, size_t padding) {
    DeviceType type = in.getDevice().get()->get_type();
    
    if(type == DeviceType::CPU){
        CpuMath::conv(in.data(),in.shape,ker.data(), ker.shape, out.data(), stride, padding);
    }else{
        GpuMath::conv(in.data(),in.shape,ker.data(), ker.shape, out.data(), stride, padding);
    }
}