#include <core/Layer.hpp>

class MaxPooling : virtual public Layer{
private:
    std::unique_ptr<Tensor> _mask;
    std::array<std::size_t, 2> window_shape;
    std::array<std::size_t, 2> stride;
    std::array<std::size_t, 2> padding;
public:
    ~MaxPooling() override = default;

    MaxPooling(
        std::array<std::size_t, 2> _window_shape,
        std::array<std::size_t, 2> _stride,
        std::array<std::size_t, 2> _padding,
        std::shared_ptr<Device> dev
    );

    std::shared_ptr<Tensor> forward( const Tensor& ) override;
    std::shared_ptr<Tensor> backward( const Tensor&, const Tensor& ) override;
    void update_weight() override;
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
        std::array<std::size_t, 2> _padding,
        std::shared_ptr<Device> dev
    );

    std::shared_ptr<Tensor> forward( const Tensor& ) override;
    std::shared_ptr<Tensor> backward( const Tensor&, const Tensor&  ) override;
    void update_weight() override;
};