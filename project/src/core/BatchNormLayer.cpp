#include <core/BatchNormLayer.hpp>
#include <random>
#include <algorithm>
#include <cassert>
#include <math/TensorMath.hpp>

BatchNorm::BatchNorm(
    std::size_t channel_dim,
    std::shared_ptr<Device> dev
) : Layer(dev),
    _gamma({channel_dim},dev,1.0f,true),
    _beta({channel_dim},dev,0.0f,true),
    _beta_grad({channel_dim},dev,0.0f,true),
    _means({channel_dim},dev,0.0f,true),
    _variances({channel_dim},dev,0.0f,true),
    _running_means({channel_dim},dev,0.0f,true),
    _running_variances({channel_dim},dev,1.0f,true),
    _sqrt_variances({channel_dim},dev,0.0f,true)
{
    //The gradient is also a vector of length channel_dim known at instantiation
    this->_grad = std::make_shared<Tensor>(std::array<std::size_t,TENSOR_MAX_DIM>{channel_dim},dev,0.0f,true);
}

std::shared_ptr<Tensor> BatchNorm::forward(
    const Tensor& in
){
    __check_initialization(this->_forward_out,in.shape);
    __check_initialization(this->_normalized_input,in.shape);
    if(mode == LayerMode::Inference){
        //Check first, if normalized input holder and forward output holder are ok


        //Computes \gamma x* + \beta
        //We need: gamma, beta, but expanded. Using unsqueeze over a copy of them
        Tensor expanded_gamma = Tensor(this->_gamma,this->_dev);
        Tensor expanded_beta = Tensor(this->_beta,this->_dev);
        Tensor expanded_means = Tensor(this->_running_means,this->_dev);

        //Before proceeding with the variances, first cache the value \sqrt(\sigma^2+\epsilon)
        Tensor epsilon = Tensor({1},this->_dev,1e-3f,true);
        TensorMath::add(
            this->_running_variances,
            epsilon,
            this->_sqrt_variances
        );
        TensorMath::sqrt(
            this->_sqrt_variances,
            this->_sqrt_variances
        );
        //Now we have computed the denominator of the normal formula. Expand that result
        Tensor expanded_sqrt_variances = Tensor(this->_sqrt_variances,this->_dev);

        //Expand correctly the variables to match input dimension
        expanded_gamma.unsqueeze(0);
        expanded_beta.unsqueeze(0);
        expanded_means.unsqueeze(0);
        expanded_sqrt_variances.unsqueeze(0);
        for(std::size_t i = 2; i < in.shape.n_dim; i++){
            expanded_gamma.unsqueeze(i);
            expanded_beta.unsqueeze(i);
            expanded_means.unsqueeze(i);
            expanded_sqrt_variances.unsqueeze(i);
        }

        //Compute x*. Note that this is stored on normalized_input
        //Start with the logic
        Tensor& forward_out = *(this->_forward_out);
        Tensor& normalized_input = *(this->_normalized_input);


        // normalized_input <- in - expanded_means (X' = X - \mu)
        TensorMath::sub(in,expanded_means,normalized_input);
        // normalized_input = normalized_input / expanded_variances  [ X' = (X' [which was = X - \mu] )/\sqrt(\sigma^2 + \epsilon) ]
        TensorMath::divide(normalized_input,expanded_sqrt_variances,normalized_input);

        //They are expanded. Now call the correct sequence of operations:
        TensorMath::multiply(expanded_gamma,normalized_input,forward_out);
        TensorMath::add(forward_out,expanded_beta,forward_out);
    }else{

        //Compute means of the incoming X
        TensorMath::mean(
            in,
            1,
            _means
        );
        //Then compute the variance
        TensorMath::variance(
            in,
            _means,
            1,
            _variances
        );
        //Start with the logic
        Tensor& forward_out = *(this->_forward_out);
        Tensor& normalized_input = *(this->_normalized_input);

        //An epsilon value
        Tensor epsilon = Tensor({1},this->_dev,1e-3f,true);

        //Cache \sqrt(\sigma^2+\epsilon) on this->_sqrt_variances: backward() reads this member,
        //so it must be persisted here (not just computed in a local expanded copy), otherwise
        //backward() divides by the stale/zero constructor value.
        TensorMath::add(
            this->_variances,
            epsilon,
            this->_sqrt_variances
        );
        TensorMath::sqrt(
            this->_sqrt_variances,
            this->_sqrt_variances
        );

        //Copy _gamma, _beta, _means, and _sqrt_variances. We don't want to modify its shape
        Tensor expanded_gamma = Tensor(this->_gamma,this->_gamma.getDevice());
        Tensor expanded_beta = Tensor(this->_beta,this->_beta.getDevice());
        Tensor expanded_means = Tensor(this->_means,this->_means.getDevice());
        Tensor expanded_variances = Tensor(this->_sqrt_variances,this->_sqrt_variances.getDevice());

        expanded_gamma.unsqueeze(0);
        expanded_beta.unsqueeze(0);
        expanded_means.unsqueeze(0);
        expanded_variances.unsqueeze(0);

        expanded_gamma.inner_expand(in.shape.n_dim);
        expanded_beta.inner_expand(in.shape.n_dim);
        expanded_means.inner_expand(in.shape.n_dim);
        expanded_variances.inner_expand(in.shape.n_dim);


        // normalized_input <- in - expanded_means (X' = X - \mu)
        TensorMath::sub(in,expanded_means,normalized_input);
        // normalized_input = normalized_input / expanded_variances  [ X' = (X' [which was = X - \mu] )/\sqrt(\sigma^2 + \epsilon) ]
        TensorMath::divide(normalized_input,expanded_variances,normalized_input);

        //They are expanded. Now call the correct sequence of operations:
        TensorMath::multiply(expanded_gamma,normalized_input,forward_out);
        TensorMath::add(forward_out,expanded_beta,forward_out);

        //Update the running means and variances
        TensorMath::scale(          //running_means = (1-momentum)running_means
            this->_running_means,
            1-momentum,
            this->_running_means
        );
        TensorMath::scale(          //means = (momentum)means
            this->_means,
            momentum,
            this->_means
        );
        TensorMath::add(            //running_means = means + running_means = (momentum)means + (1-momentum) running_means
            this->_running_means,
            this->_means,
            this->running_means
        );
        //Same logic for variance
        TensorMath::scale(
            this->_running_variances,
            1-momentum,
            this->_running_variances
        );
        TensorMath::scale(
            this->_variances,
            momentum,
            this->_variances
        );
        TensorMath::add(
            this->_running_variances,
            this->_variances,
            this->_running_variances
        );

    }

    return this->_forward_out;
}

std::shared_ptr<Tensor> BatchNorm::backward( const Tensor& backward_grad, const Tensor& forward_input  ){
    assert(_normalized_input.get() != nullptr);
    assert( backward_grad.shape == forward_input.shape );

    std::size_t n_dim = backward_grad.shape.n_dim;
    std::size_t channel_dim = backward_grad.shape[1];
    std::size_t total_elements = backward_grad.shape.n_elements;
    __check_initialization(this->_grad, Shape({channel_dim}));
    __check_initialization(this->_backward_out,backward_grad.shape);
    //Compute \beta gradient
    Tensor& _gamma_grad = *(this->_grad);
    Tensor& normalized_input = *(this->_normalized_input);
    TensorMath::reduce_all(
        backward_grad,
        this->_beta_grad,
        1       //Channel axis
    );
    //Compute \gamma gradient
    TensorMath::multiply(
        backward_grad,
        normalized_input,
        *(this->_backward_out)     //Warning: using _backward_out as a temporary value holder
    );

    TensorMath::reduce_all(
        *(this->_backward_out),
        _gamma_grad,
        1                   //Over channel axis
    );

    //Now compute the gradient to pass backwards
    //To do so, we need to expand the gradients to match the expected layout

    //Now that every tensor has the same dimension, we proceed
    //compute dL/dX = (\gamma / M) . 1/\sqrt(\sigma^2 + \epsilon) . (M . dL/dY . dL/d\beta - X_normalized . dL/d\gamma )
    float M = total_elements/channel_dim;
    //Create copies to expand in the formula
    //In the formula appears and needs to be expanded: gamma, _sqrt_variances, _beta_grad, _gamma_grad
    Tensor expanded_grad_gamma = Tensor(_gamma_grad,this->_dev);
    Tensor expanded_grad_beta = Tensor(_beta_grad,this->_dev);
    Tensor expanded_sqrt_variances = Tensor(_sqrt_variances,this->_dev);
    Tensor expanded_gamma = Tensor(_gamma,this->_dev);

    expanded_grad_beta.unsqueeze(0);
    expanded_grad_gamma.unsqueeze(0),
    expanded_sqrt_variances.unsqueeze(0);
    expanded_gamma.unsqueeze(0);
    expanded_grad_beta.inner_expand(n_dim);
    expanded_grad_gamma.inner_expand(n_dim),
    expanded_sqrt_variances.inner_expand(n_dim);
    expanded_gamma.inner_expand(n_dim);

    //Now that all are well expanded, compute (M . dL/dY - dL/d\beta - X_normalized . dL/d\gamma )
    //In two steps. First M . dL/dY - dL/d\beta saved in _backward_out, and then X_normalized . dL/d\gamma saved in X_normalized
    // I believe X_normalized is not needed in further procedures after backwards
    TensorMath::scale(
        backward_grad,
        M,
        *(this->_backward_out)
    );
    TensorMath::sub(
        *(this->_backward_out),
        expanded_grad_beta,
        *(this->_backward_out)
    );
    TensorMath::multiply(
        normalized_input,
        expanded_grad_gamma,
        normalized_input
    );
    TensorMath::sub(
        *(this->_backward_out),
        normalized_input,
        *(this->_backward_out)
    );
    //Now compute the rest of the formula
    TensorMath::divide(
        *(this->_backward_out),
        expanded_sqrt_variances,
        *(this->_backward_out)
    );
    TensorMath::multiply(
        *(this->_backward_out),
        expanded_gamma,
        *(this->_backward_out)
    );
    TensorMath::scale(
        *(this->_backward_out),
        1/M,
        *(this->_backward_out)
    );
    return this->_backward_out;
}

void BatchNorm::update_weight(float lr){
    assert(this->_grad.get() != nullptr);
    //This is quite simple, just add _grad*lr to _gamma and _beta_grad*lr to _beta
    TensorMath::scale(
        *(this->_grad),
        lr,
        *(this->_grad)
    );
    TensorMath::add(
        *(this->_grad),
        _gamma,
        _gamma
    );
    TensorMath::scale(
        this->_beta_grad,
        lr,
        this->_beta_grad
    );
    TensorMath::add(
        this->_beta_grad,
        _beta,
        _beta
    );

}
