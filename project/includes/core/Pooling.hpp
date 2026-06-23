#include <core/Layer.hpp>

class MaxPooling : virtual public Layer{
private:
    std::array<std::size_t, 2> window_shape;
    std::array<std::size_t, 2> stride;
    std::array<std::size_t, 2> padding;
public:
    ~MaxPooling() override = default;

    MaxPooling(
        std::array<std::size_t, 2> _window_shape,
        std::array<std::size_t, 2> _stride,
        std::array<std::size_t, 2> _padding
    ) : window_shape(_window_shape),stride(_stride),padding(_padding){}

    Tensor forward(Tensor&) override;
    void update_weights(Tensor&) override;
};  

class AvgPooling : public Layer{
private:
    std::array<std::size_t, 2> window_shape;
    std::array<std::size_t, 2> stride;
    std::array<std::size_t, 2> padding;
public:
    ~AvgPooling() override = default;

    AvgPooling(
        std::array<std::size_t, 2> _window_shape,
        std::array<std::size_t, 2> _stride,
        std::array<std::size_t, 2> _padding
    ) : window_shape(_window_shape),stride(_stride),padding(_padding){}

    Tensor forward(Tensor&) override;
    void update_weights(Tensor&) override;
};