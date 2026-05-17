#include <cnn/tensor.hpp>

class Layer{
protected:
    virtual Tensor forward(Tensor);
    virtual void update_weights(Tensor);
    
};


class Pooling : public Layer{
protected:
    int pool_size;
    int stride;
    virtual float pool_patch();
    virtual void backprop_patch();

public:
    Tensor forward(Tensor) override;
    void update_weights(Tensor) override;

};
