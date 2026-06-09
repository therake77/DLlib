#pragma once
#include <core/Shape.hpp>

class Math{
public:
    virtual ~Math() = default;
    Math() = default;

    virtual void add(const float* a, const float* b, float* out, size_t size) = 0;
    virtual void sub(const float* a, const float* b, float* out, size_t size) = 0;
    virtual void scale(const float* a, float* out, float _n, size_t size) = 0;
    virtual void multiply(const float* a, const float* b, float* out, size_t size) = 0;

    //Linear Algebra
    virtual void matmul(
        const float* a , Shape a_shape,  
        const float* b , Shape b_shape,
        float* out, Shape out_shape
    ) = 0;
    
    virtual void conv2D(
        const float* in, Shape in_shape,
        const float* ker, Shape ker_shape,
        float* out, Shape out_shape,
        std::array<size_t,2> stride, std::array<size_t,2> padding
    ) = 0;

    virtual void conv3D(
        const float* in, Shape in_shape,
        const float* ker, Shape ker_shape,
        float* out, Shape out_shape,
        std::array<size_t,3> stride, std::array<size_t,3> padding
    ) = 0;

    //Activation functions
    virtual void reLU(
        const float* a, float* out, size_t n_elements
    ) = 0;
    virtual void sigmoid(const float* a, float* out, size_t n_elements) = 0;
    virtual void tanh(const float* a, float* out, size_t n_elements) = 0;
    virtual void softmax(const float* a, float* out, size_t n_elements) = 0;
    
    //Structural ops and reshaping
    virtual void sum(
        const float* a, Shape a_shape, 
        float* out, Shape out_shape ,
        size_t axis
    ) = 0;
};