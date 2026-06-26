#pragma once
#include <math/Math.hpp>

class CpuMath : public Math{
public:
    ~CpuMath() override = default;
    CpuMath() = default;

    void add(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) override;
    void sub(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) override;
    void scale(const float* a, float* out, float _n, Shape shape) override;
    void multiply(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) override;
    void divide(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) override;
    void sqrt(const float* a,  Shape a_shape, float* out, Shape out_shape) override;
    void pow(const float* a,  Shape a_shape, int _n, float* out, Shape out_shape) override;
    
    void matmul(
        const float* a , Shape a_shape,  
        const float* b , Shape b_shape,
        float* out, Shape out_shape
    ) override;
    
    void conv2D(
        const float* in, Shape in_shape,
        const float* ker, Shape ker_shape,
        float* out, Shape out_shape,
        std::array<size_t,2> stride, std::array<size_t,2> padding
    ) override;

    void reLU(const float* a, float* out, size_t n_elements) override;
    void sigmoid(const float* a, float* out, size_t n_elements) override;
    void tanh(const float* a, float* out, size_t n_elements) override;
    void softmax(const float* a, float* out, size_t n_elements) override;
    void sum(
        const float* a, Shape a_shape, 
        float* out, Shape out_shape ,
        size_t axis
    ) override;

    //Pooling operations
    void max_pooling2D(
        const float* input,
        const Shape input_shape,
        std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes,    //(Height , Width)
        float* mask,
        const Shape mask_shape,
        float* out,
        const Shape out_shape,
        std::array<std::size_t, 2> padding
    ) override;

    void inv_max_pooling2D(
        const float* max_pooled,
        const Shape max_pooled_shape,
        std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes,    //(Height , Width)
        const float* mask,
        const Shape mask_shape,
        float* out,
        const Shape out_shape,
        std::array<std::size_t, 2> padding
    ) override;

    void avg_pooling2D(
        const float* input,
        const Shape input_shape,
        std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes,    //(Height , Width)
        float* out,
        const Shape out_shape,
        std::array<std::size_t, 2> padding
    ) override;

    void inv_avg_pooling2D(
        const float* input,
        const Shape input_shape,
        std::array<std::size_t,2> strides,  
        std::array<std::size_t,2> window_shapes,    //(Height , Width)
        float* out,
        const Shape out_shape,
        std::array<std::size_t, 2> padding
    ) override;

    void transpose(
        const float* in, const Shape& in_shape,
        float* out, const Shape& out_shape
    ) override;

    void binary_positive_mask(
        const float* in,
        const Shape& in_shape,
        float* out,
        const Shape& out_shape
    ) override;

    void mean(
        const float* in,
        const Shape& in_shape,
        float* out,
        const Shape& out_shape,
        std::size_t axis
    ) override;

    void variance(
        const float* in,
        const Shape& in_shape,
        const float* mean,
        const Shape& mean_shape,
        float* out,
        const Shape& out_shape,
        std::size_t axis
    ) override;

    void reduce_all(
        const float* in,
        const Shape& in_shape,
        float* out,
        const Shape& out_shape,
        std::size_t axis
    ) override;

private:
    void _transpose2D(
        const float* in, const Shape& in_shape,
        float* out, const Shape& out_shape
    );
};
