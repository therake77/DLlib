#include <core/BatchNormLayer.hpp>
#include <random>
#include <algorithm>
#include <cassert>
#include <math/TensorMath.hpp>

BatchNorm::BatchNorm(
    std::size_t channel_dim,
    std::shared_ptr<Device> dev
) : Layer({1,channel_dim,1,1},dev),
    _gamma({channel_dim},dev),
    _beta({channel_dim},dev),
    _means({channel_dim},dev),
    _variances({channel_dim},dev)
{}

Tensor& BatchNorm::forward(
    Tensor& input
){
    //First, has _result been correctly allocated?
    if(this->_result.shape != input.shape){
        this->_result.resize(input.shape);
    }

    //Computes \gamma x* + \beta
    //We need: gamma, beta, but expanded. Using unsqueeze and leaving intact the original _gamma and _betas
    Tensor expanded_gamma = Tensor(this->_gamma,this->_gamma.getDevice());
    Tensor expanded_beta = Tensor(this->_beta,this->_beta.getDevice());
    Tensor expanded_means = Tensor(this->_means,this->_means.getDevice());
    Tensor expanded_variances = Tensor(this->_variances,this->_variances.getDevice());
    
    

    expanded_gamma.unsqueeze(0);
    expanded_beta.unsqueeze(0);
    expanded_means.unsqueeze(0);
    expanded_variances.unsqueeze(0);
    for(std::size_t i = 2; i < input.shape.n_dim; i++){
        expanded_gamma.unsqueeze(i);
        expanded_beta.unsqueeze(i);
        expanded_means.unsqueeze(i);
        expanded_variances.unsqueeze(i);
    }

    //Compute x*
    TensorMath::sub(input,expanded_means,this->_result);    //result = input - expanded_means
    TensorMath::sqrt(expanded_variances,expanded_variances);    //expanded_variances = \sqrt(expanded_variances)
    TensorMath::divide(this->_result,expanded_variances,this->_result); //result = (input - expanded_means)/\sqrt(expanded_variances)

    //They are expanded. Now call the correct sequence of operations:
    TensorMath::multiply(expanded_gamma,this->_result,this->_result);
    TensorMath::add(this->_result,expanded_beta,this->_result);

    return this->_result;
}

void update_weights(Tensor&){
    throw std::runtime_error("Not implemented");
}

Tensor& BatchNorm::forward(Tensor& input , bool training){
    if(!training){
        return this->forward(input);
    }else{
        throw std::runtime_error("Not implemented");
    }
}