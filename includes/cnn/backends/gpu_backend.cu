#include <cnn/tensor.hpp>

namespace GPU{

    Tensor add(Tensor& a, Tensor& b);
    Tensor scalarMultiply(Tensor& a);
    Tensor multipy(Tensor& a, Tensor& b);

    float* allocateMemory(const std::vector<float>&);
    void freeMemory(const float*);
    float* getFromMemory(float *);

};