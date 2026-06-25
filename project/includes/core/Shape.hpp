#pragma once
#include <array>

#define TENSOR_MAX_DIM 4

/* Define las dimensiones del tensor */
class Shape{
private:
    void _trigger_shape_recomposition();

public:
    /* Número de dimensiones del tensor (desde 1 hasta `TENSOR_MAX_DIM`) */
    std::size_t n_dim;
    /* Total de números (floats) que almacena el tensor */
    std::size_t n_elements;
    std::array<std::size_t,TENSOR_MAX_DIM> _shapes;
    std::array<std::size_t, TENSOR_MAX_DIM> _strides;
    Shape()= delete;
    Shape(std::array<size_t, TENSOR_MAX_DIM>);
    const size_t& operator[](std::size_t index) const;
    size_t& operator[](std::size_t index);
    bool operator==(const Shape&) const;
    bool operator!=(const Shape&) const;

    Shape unsqueeze(std::size_t);
    static void broadcast_shapes(Shape& a, Shape& b);
};