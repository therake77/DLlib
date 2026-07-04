#pragma once

#include <data/transforms/Transform.hpp>

#include <initializer_list>
#include <memory>
#include <vector>

class Compose : public Transform {
public:
    Compose() = default;
    explicit Compose(std::vector<std::shared_ptr<Transform>> transforms);
    Compose(std::initializer_list<std::shared_ptr<Transform>> transforms);

    Tensor apply(const Tensor& input) const override;
    Tensor operator()(const Tensor& input) const;

    void add(std::shared_ptr<Transform> transform);
    std::size_t size() const { return _transforms.size(); }
    bool empty() const { return _transforms.empty(); }

private:
    std::vector<std::shared_ptr<Transform>> _transforms;
};
