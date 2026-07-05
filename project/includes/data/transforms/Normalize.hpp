#pragma once

#include <data/transforms/Transform.hpp>

#include <cstddef>
#include <vector>

class Normalize : public Transform {
public:
    Normalize(float mean, float std);
    Normalize(std::vector<float> mean, std::vector<float> std);

    Tensor apply(const Tensor& input) const override;
    Tensor operator()(const Tensor& input) const;

    const std::vector<float>& mean() const { return _mean; }
    const std::vector<float>& std() const { return _std; }

private:
    std::vector<float> _mean;
    std::vector<float> _std;

    static float _channel_value(
        const std::vector<float>& values,
        std::size_t channel
    );
};
