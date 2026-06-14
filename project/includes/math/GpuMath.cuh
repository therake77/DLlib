#include <math/Math.hpp>

class GpuMath : public Math{
public:

    ~GpuMath() override = default;
    GpuMath() = default;

    void add(const float* a, const float* b, float* out, size_t size) override;
    void sub(const float* a, const float* b, float* out, size_t size) override;
    void scale(const float* a, float* out, float _n, size_t size) override;
    void multiply(const float* a, const float* b, float* out, size_t size) override;
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
    void conv3D(
        const float* in, Shape in_shape,
        const float* ker, Shape ker_shape,
        float* out, Shape out_shape,
        std::array<size_t,3> stride, std::array<size_t,3> padding
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

};