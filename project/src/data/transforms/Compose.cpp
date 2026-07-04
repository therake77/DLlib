#include <data/transforms/Compose.hpp>

#include <stdexcept>
#include <utility>

Compose::Compose(std::vector<std::shared_ptr<Transform>> transforms)
    : _transforms(std::move(transforms)) {
    for(const auto& transform : _transforms) {
        if(!transform) {
            throw std::runtime_error("Error: Compose received a null transform");
        }
    }
}

Compose::Compose(std::initializer_list<std::shared_ptr<Transform>> transforms)
    : Compose(std::vector<std::shared_ptr<Transform>>(transforms)) {
}

Tensor Compose::operator()(const Tensor& input) const {
    return apply(input);
}

Tensor Compose::apply(const Tensor& input) const {
    std::shared_ptr<Tensor> output =
        std::make_shared<Tensor>(input, input.getDevice());

    for(const auto& transform : _transforms) {
        Tensor next = transform->apply(*output);
        output = std::make_shared<Tensor>(next, next.getDevice());
    }

    return Tensor(*output, output->getDevice());
}

void Compose::add(std::shared_ptr<Transform> transform) {
    if(!transform) {
        throw std::runtime_error("Error: Cannot add a null transform to Compose");
    }

    _transforms.push_back(std::move(transform));
}
