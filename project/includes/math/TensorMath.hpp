#include <core/Tensor.hpp>
#include <functional>

namespace TensorMath {
    void add(const Tensor& a, const Tensor&b, Tensor& out);
    void sub(const Tensor& a, const Tensor&b, Tensor& out);
    void scale( const Tensor& a, float n, Tensor& out);
    void multiply(const Tensor& a, const Tensor&b, Tensor& out);
    void divide(const Tensor& a, const Tensor&b, Tensor& out);
    void sqrt(const Tensor& a, Tensor& out);
    void pow(const Tensor& a, const int i, Tensor& out);
    void log(const Tensor& a, Tensor& out);
    

    void matmul(const Tensor& a, const Tensor&b, Tensor& out);
    void conv2D(const Tensor& in, const Tensor& ker, Tensor& out, size_t stride, size_t padding);
    void conv2D(const Tensor& in, const Tensor& ker, Tensor& out, std::array<size_t,2> stride, std::array<size_t,2> padding);
    
    void im2col2D(
        const Tensor& in, 
        std::array<std::size_t,2> window_shape,
        std::array<std::size_t,2> strides,
        std::array<std::size_t,2> padding,
        Tensor& out
    );

    void col2im2D(
        const Tensor& in, 
        std::array<std::size_t,2> window_shape,
        std::array<std::size_t,2> strides,
        std::array<std::size_t,2> padding,
        Tensor& out
    );

    void reLU(const Tensor& a, Tensor& out);
    void sigmoid(const Tensor& a, Tensor& out);
    void tanh(const Tensor& a, Tensor& out);
    void softmax(const Tensor& a, Tensor& out);

    //I'm thinking of removing this
    void sum(const Tensor& a, size_t axis, Tensor& out);
    Tensor reshape(const Tensor& a, std::array<size_t, TENSOR_MAX_DIM> new_shape);
    Tensor transpose(const Tensor& a);

    void max_pooling2d(
        const Tensor& in, Tensor& out, Tensor& mask, std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes, 
        std::array<std::size_t, 2> padding
    );

    void inv_max_pooling2D(
        const Tensor& max_pooled,
        std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes,    //(Height , Width)
        const Tensor& mask,
        Tensor& out,
        std::array<std::size_t, 2> padding
    );

    void avg_pooling2d(
        const Tensor& in, Tensor& out, std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes, 
        std::array<std::size_t, 2> padding
    );

    void inv_avg_pooling2D(
        const Tensor& avg_pooled,
        std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes,    //(Height , Width)
        Tensor& out,
        std::array<std::size_t, 2> padding
    );

    void binary_positive_mask(
        const Tensor& in,
        Tensor& out
    );

    void mean(
        const Tensor& in,
        std::size_t axis,
        Tensor& out
    );

    void variance(
        const Tensor& in,
        const Tensor& means,
        std::size_t axis,
        Tensor& out
    );

    void reduce_all(
        const Tensor& in,
        Tensor& out,
        std::size_t axis
    );

};