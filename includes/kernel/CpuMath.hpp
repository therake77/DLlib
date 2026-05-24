#include <core/Tensor.hpp>

namespace CpuMath{

    void add(const float* a, const float* b, size_t size, float* out);
    void sub(const float* a, const float* b, size_t size, float* out);
    void scale(const float* a, size_t size, float _n, float* out);
    void multiply(const float* a, const float* b, size_t size, float* out);

    //Linear Algebra
    void matmul(const float* a, std::vector<size_t> a_shape, const float* b, std::vector<size_t> b_shape, float* out);
    //Still researching about it
    void conv(const float* in, std::vector<size_t> in_shape, const float* ker, std::vector<size_t> ker_shape, float* out, size_t stride, size_t padding);

    //Activation functions
    void reLU(const float* a, size_t size, float* out);
    void sigmoid(const float* a, size_t size, float* out);
    void tanh(const float* a, size_t size, float* out);
    void softmax(const float* a, size_t size, float* out);
    
    //Structural ops and reshaping
    void sum(const float* a, std::vector<size_t> shape, size_t axis, float* out);
    //TODO(Do we really need a reshape at this level?)
    void reshape(const float* a, std::vector<size_t> shape, std::vector<size_t> new_shape, float* out);
    void transpose(const float* a, std::vector<size_t> shape, float* out);

};
