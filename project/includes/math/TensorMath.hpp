#include <core/Tensor.hpp>

namespace TensorMath {
    void add(Tensor& a, Tensor&b, Tensor& out);
    void sub(Tensor& a, Tensor&b, Tensor& out);
    void scale(Tensor& a, float n, Tensor& out);
    void multiply(Tensor& a, Tensor& b, Tensor& out);

    void matmul(Tensor& a , Tensor& b , Tensor& out);
    void conv2D(Tensor& in, Tensor& ker, Tensor& out, size_t stride, size_t padding);
    void conv3D(Tensor& in, Tensor& ker, Tensor& out, size_t stride, size_t padding);

    void reLU(Tensor& a, Tensor& out);
    void sigmoid(Tensor& a, Tensor& out);
    void tanh(Tensor& a, Tensor& out);
    void softmax(Tensor& a, Tensor& out);

    void sum(Tensor& a, size_t axis, Tensor& out);
    Tensor reshape(Tensor& a, std::array<size_t, TENSOR_MAX_DIM> new_shape);
    Tensor transpose(Tensor& a);
};