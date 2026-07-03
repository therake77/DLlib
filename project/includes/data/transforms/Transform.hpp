#pragma once

#include <core/Tensor.hpp>

class Transform {
public:
    virtual ~Transform() = default;

    virtual Tensor apply(const Tensor& input) const = 0;
};
