#include <core/Layer.hpp>

class Conv2D : public Layer{

    Tensor forward(Tensor) override;
    void update_weights(Tensor) override;

};

class FullyConnected : public Layer{
    
    Tensor forward(Tensor) override;
    void update_weights(Tensor) override;

};
