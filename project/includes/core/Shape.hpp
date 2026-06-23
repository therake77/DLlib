#pragma once
#include <array>

#define TENSOR_MAX_DIM 4

/* Define las dimensiones del tensor */
class Shape{
public:
    /* Número de dimensiones del tensor (desde 1 hasta `TENSOR_MAX_DIM`) */
    size_t n_dim;
    /* Total de números (floats) que almacena el tensor */
    size_t n_elements;
    std::array<size_t,TENSOR_MAX_DIM> _shapes;
    std::array<size_t, TENSOR_MAX_DIM> _strides;
    Shape()= delete;
    Shape(std::array<size_t, TENSOR_MAX_DIM>);
    const size_t& operator[](size_t index) const;
    size_t& operator[](size_t index);
    bool operator==(const Shape&) const;
    bool operator!=(const Shape&) const;
};