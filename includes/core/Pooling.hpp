#include <core/Layer.hpp>


class MaxPooling : public Pooling{
    float pool_patch() override;
    void backprop_patch() override;
};  

class AvgPooling : public Pooling{
    float pool_patch() override;    
    void backprop_patch() override;
};