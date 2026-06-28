#include <core/Layer.hpp>

class OutputLayer : public Layer{
private:

    Tensor _labels;
    Tensor _probabilities;
    std::size_t _logits;
public:
    OutputLayer(
        std::shared_ptr<Device> dev,
        std::size_t logits
    );
    ~OutputLayer() override = default;

    std::shared_ptr<Tensor> forward( const Tensor& ) override;
    std::shared_ptr<Tensor> backward( const Tensor&, const Tensor& ) override;
    void update_weight() override;
    void set_labels(Tensor& labels);
};