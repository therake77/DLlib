#include <core/Layer.hpp>


class MaxPooling : public Pooling{
public:
    ~MaxPooling() override = default;
    float pool_patch() override;
    void backprop_patch() override;
};  

class AvgPooling : public Pooling{
public:
    ~AvgPooling() override = default;
    float pool_patch() override;    
    void backprop_patch() override;
};