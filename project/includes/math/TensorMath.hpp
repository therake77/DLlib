#include <core/Tensor.hpp>

namespace TensorMath {
    void add(Tensor& a, Tensor&b, Tensor& out);
    void sub(Tensor& a, Tensor&b, Tensor& out);
    void scale(Tensor& a, float n, Tensor& out);
    void multiply(Tensor& a, Tensor& b, Tensor& out);

    void matmul(Tensor& a , Tensor& b , Tensor& out);
    void conv2D(Tensor& in, Tensor& ker, Tensor& out, size_t stride, size_t padding);
    void conv3D(Tensor& in, Tensor& ker, Tensor& out, size_t stride, size_t padding);

    void conv2D(Tensor& in, Tensor& ker, Tensor& out, std::array<size_t,2> stride, std::array<size_t,2> padding);
    void conv3D(Tensor& in, Tensor& ker, Tensor& out, std::array<size_t,3> stride, std::array<size_t,3> padding);
    

    void reLU(Tensor& a, Tensor& out);
    void sigmoid(Tensor& a, Tensor& out);
    void tanh(Tensor& a, Tensor& out);
    void softmax(Tensor& a, Tensor& out);

    void sum(Tensor& a, size_t axis, Tensor& out);
    Tensor reshape(Tensor& a, std::array<size_t, TENSOR_MAX_DIM> new_shape);
    Tensor transpose(Tensor& a);

    void max_pooling2d(
        Tensor& in, Tensor& out, std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes, 
        std::array<std::size_t, 2> padding
    );

    void avg_pooling2d(
        Tensor& in, Tensor& out, std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes, 
        std::array<std::size_t, 2> padding
    );

};