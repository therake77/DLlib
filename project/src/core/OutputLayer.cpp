#include <core/OutputLayer.hpp>
#include <math/TensorMath.hpp>
#include <cassert>

OutputLayer::OutputLayer(
    std::shared_ptr<Device> dev,
    std::size_t logits
) : _logits(logits) , Layer(dev), _labels({1,logits},dev), _probabilities({1,logits}) { }

std::shared_ptr<Tensor> OutputLayer::forward( const Tensor& in ){
    
    if( mode == LayerMode::Inference){
        __check_initialization(this->_forward_out,in.shape);

        TensorMath::softmax(
            in,
            *(this->_forward_out)
        );

        return this->_forward_out;
    }else{
        //Training
        std::size_t batchs = in.shape[0]; 
        __check_initialization(this->_forward_out,in.shape);
        _probabilities.resize(in.shape);
        TensorMath::softmax(
            in,
            this->_probabilities  //Use grad to save the softmax result
        );
        TensorMath::log(
            this->_probabilities, //And then, use _forward_out to compute the rest
            *(this->_forward_out)
        );
        TensorMath::multiply(
            *(this->_forward_out),
            this->_labels,
            *(this->_forward_out)
        );
        __check_initialization(this->_grad,Shape({batchs}));
        this->_grad->zeros();
        TensorMath::sum(
            *(this->_forward_out),
            1,
            *(this->_grad)
        );
        this->_forward_out->resize(Shape({1}));
        TensorMath::sum(
            *(this->_grad),
            0,
            *(this->_forward_out)
        );
        TensorMath::scale(
            *(this->_forward_out),
            -1/batchs,
            *(this->_forward_out)
        );
        return this->_forward_out;
    }
}

//Important: backward_grad in this layer is the real labels
std::shared_ptr<Tensor> OutputLayer::backward( const Tensor& backward_grad, const Tensor& forward_in ) {

    assert(this->_forward_out.get() != nullptr);
    assert(forward_in.shape.n_dim == 2);
    std::size_t batch = forward_in.shape[0];
    __check_initialization(this->_backward_out,forward_in.shape);

    TensorMath::sub(
        *(this->_forward_out),
        backward_grad,
        *(this->_backward_out)
    );
    TensorMath::scale(
        *(this->_backward_out),
        1/batch,
        *(this->_backward_out)
    );
    return this->_backward_out;
}

void OutputLayer::update_weight(){
    //No weights
    return;
}

void OutputLayer::set_labels(Tensor& labels){
    this->_labels.resize(labels.shape);
    this->_labels.copy_from(labels);
    return;
}