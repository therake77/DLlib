#include <core/Tensor.hpp>


//This class requires some level of cache or memory management. So is just for the sake of testing
// And to keep a clean API
namespace TensorMath{
    //Element-wise operations
    Tensor add(Tensor& a, Tensor&b);
    Tensor sub(Tensor& a, Tensor&b);
    Tensor scale(Tensor& a,float n);
    Tensor multiply(Tensor& a, Tensor& b);

    //Linear Algebra
    Tensor matmul(Tensor& a , Tensor& b);
    Tensor conv(Tensor& in, Tensor& ker, size_t stride, size_t padding);

    //Activation functions
    Tensor reLU(Tensor& a);
    Tensor sigmoid(Tensor& a);
    Tensor tanh(Tensor& a);
    Tensor softmax(Tensor& a);

    //Structural ops and reshaping
    Tensor sum(Tensor& a, size_t axis);
    Tensor reshape(Tensor& a, std::vector<size_t> shape);
    Tensor transpose(Tensor& a);

};

//Low, high-performance API
namespace BaseMath{
    //Element-wise operations
    void add(Tensor& a, Tensor&b, Tensor& out);
    void sub(Tensor& a, Tensor&b, Tensor& out);
    void scale(Tensor& a, float n, Tensor& out);
    void multiply(Tensor& a, Tensor& b, Tensor& out);

    //Linear Algebra
    void matmul(Tensor& a , Tensor& b , Tensor& out);
    void conv(Tensor& in, Tensor& ker, Tensor& out, size_t stride, size_t padding);

    //Activation functions
    void reLU(Tensor& a, Tensor& out);
    void sigmoid(Tensor& a, Tensor& out);
    void tanh(Tensor& a, Tensor& out);
    void softmax(Tensor& a, Tensor& out);
    
    //Structural ops and reshaping
    void sum(Tensor& a, size_t axis, Tensor& out);
    void reshape(Tensor& a, std::vector<size_t> shape, Tensor& out);
    void transpose(Tensor& a, Tensor& out);

};