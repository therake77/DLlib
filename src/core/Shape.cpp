#include <core/Shape.hpp>
#include <stdexcept>

/* Please, don't put something like {5,...,0,...,0,...,5}. it will break the logic  */
Shape::Shape(std::array<size_t,TENSOR_MAX_DIM> arr) : _shapes(arr),_strides({0}) {
    
    size_t index = TENSOR_MAX_DIM - 1;
    while(arr[index] == 0){ index--; }
    if(index >= TENSOR_MAX_DIM ) { throw std::runtime_error("Error: Tensor with zero dimensions not allowed"); }
    
    size_t _n_dims = index + 1;
    this->_strides[index] = 1;
    size_t _n_elems = arr[index];
    
    for(size_t i = index; i --> 0; ){
        _n_elems*=arr[i];
        this->_strides[i] =  arr[i+1] * this->_strides[i+1];
    }
    
    this->n_elements = _n_elems;
    this->n_dim = _n_dims;

}

bool Shape::operator==(const Shape& other) const{
    return this->_shapes == other._shapes;
}

bool Shape::operator!=(const Shape& other) const{
    return this->_shapes != other._shapes;
}

const size_t& Shape::operator[](size_t index) const{ 
    if(index < 0 || index >= TENSOR_MAX_DIM){ throw std::runtime_error("Index out of bounds"); }
    return this->_shapes[index]; 
}

size_t& Shape::operator[](size_t index){
    if(index < 0 || index >= TENSOR_MAX_DIM){ throw std::runtime_error("Index out of bounds"); }
    return this->_shapes[index]; 
}