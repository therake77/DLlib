#include <core/Tensor.hpp>

namespace CpuMath{

    void add(float* a, float* b, size_t size, float* out);
    void sub(float* a, float* b, size_t size, float* out);
    void scale(float* a, size_t size, float _n, float* out);
    void multiply(float* a, float* b, size_t size, float* out);

    //Linear Algebra
    void matmul(float* a, std::vector<size_t> a_shape, float* b, std::vector<size_t> b_shape, float* out);
    //Still researching about it
    void conv(float* in, std::vector<size_t> in_shape, float* ker, std::vector<size_t> ker_shape, float* out, size_t stride, size_t padding);

    //Activation functions
    void reLU(float* a, size_t size, float* out);
    void sigmoid(float* a, size_t size, float* out);
    void tanh(float* a, size_t size, float* out);
    void softmax(float* a, size_t size, float* out);
    
    //Structural ops and reshaping
    void sum(float* a, std::vector<size_t> shape, size_t axis, float* out);
    //TODO(Do we really need a reshape at this level?)
    void reshape(float* a, std::vector<size_t> shape, std::vector<size_t> new_shape, float* out);
    void transpose(float* a, std::vector<size_t> shape, float* out);

};
