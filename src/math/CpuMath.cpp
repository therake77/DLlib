#include <stdexcept>
#include <cmath>
#include <math/CpuMath.hpp>
#include <vector>
#include <cassert>
#include <string>

namespace{
    /* Probably would be use for something in the future*/
    bool _is_contiguous(const std::array<size_t,TENSOR_MAX_DIM> _strides){
        size_t expected_stride = 1;
        for(size_t i = _strides.size() - 1; i >= 0; --i ){
            if(_strides[i] != expected_stride){ return false; }
            expected_stride*=_strides[i];
        }
        return true;
    }

    void _standard_2d_matrix_mul(
        const float* a, const float* b, float* out, 
        size_t M, size_t N, size_t K,
        size_t stride_am, size_t stride_ak,
        size_t stride_bk, size_t stride_bn,
        size_t stride_outm, size_t stride_outn
    ){
        for(int m = 0; m < M; ++m){
            for(int n = 0; n < N; ++n){
                float sum = 0.0f;
                for(int k = 0; k < K; ++k){
                    int a_idx = (m * stride_am) + (k * stride_ak);
                    int b_idx = (k * stride_bk) + (n * stride_bn);
                    sum += a[a_idx]*b[b_idx];
                }
                int out_idx = (m * stride_outm) + (n * stride_outn);
                out[out_idx] = sum;
            }
        }
    }

    /*
    Given the shape of a kernel, transforms the input in a new matrix such that each possible kernel window looks as a row
    Expects: 
    The input with shape (Batch ,Channel ,S_0,...,S_n) with S_i spatial dimensions of the convolution
    A kernel of shape (C_out, Channel, S'_0, ... , S'_n) with S'_i spatial dimensions of the kernel 
    A output of shape ()
    */
    void im2col(
        const float* input,
        const Shape& input_shape,
        const Shape& kernel_shape,
        const std::vector<int>& padding,
        const std::vector<int>& stride,
        float* output,
        const Shape& out_shape
    ){
        assert( input_shape.n_dim == kernel_shape.n_dim && input_shape.n_dim == padding.size() && input_shape.n_dim == stride.size() );
        size_t input_ndims = input_shape.n_dim;
        if( input_ndims < 2 ){ throw std::runtime_error("Error: The input is 1D or less"); }
        size_t spatial_dims = input_ndims - 2;
        size_t batch = input_shape._shapes[0];
        size_t channels = input_shape._shapes[1];

        //Validation
        if(kernel_shape._shapes[1] != channels){ throw std::runtime_error("Error: Kernel input channel dim must match image channel dim"); }
        if(out_shape._shapes[0] != batch){ throw std::runtime_error("Error: Output batch size must match input batch size"); }
        if(out_shape._shapes[1] != kernel_shape._shapes[0]){ throw std::runtime_error("Output channel dim must match kernel output channel dim"); }

        size_t total_out_spatial_elements = 1;
        size_t total_kernel_spatial_elements = 1;
        for(size_t i = 0; i < spatial_dims; ++i){
            size_t k_spatial_size = kernel_shape._shapes[2+i];
            size_t input_spatial_size = input_shape._shapes[2+i];
            size_t expected = (input_spatial_size + 2* padding[i] - k_spatial_size)/stride[i] +  1;
            if(out_shape._shapes[2+i] != expected){ throw std::runtime_error("Error: Unexpected output dimension"); }

            total_out_spatial_elements *= out_shape._shapes[2+i];
            total_kernel_spatial_elements*=k_spatial_size;
        }
        
    }

};


void CpuMath::add(const float* a, const float* b, float* out, size_t size){
    for(size_t i = 0; i < size; i++){
        out[i] = a[i] + b[i];
    }
}

void CpuMath::sub(const float* a, const float* b, float* out, size_t size) {

    for(size_t i = 0; i < size; i++){
        out[i] = a[i] - b[i];
    }
}

void CpuMath::scale(const float* a, float* out, float _n, size_t size) {
    for(size_t i = 0; i < size; i++){
        out[i] = _n*a[i];
    }
}

void CpuMath::multiply(const float* a, const float* b, float* out, size_t size) {
    for(size_t i = 0; i < size; i++){
        out[i] = a[i] * b[i];
    }
}

/* Contracts two tensors on the two innermost dimensions
Expects: Two tensors, of shape (A,B,C,...,M,K) and (A,B,C,...,K,N)
Input tensors must have been broadcasted up to TENSOR_MAX_DIM, and output tensor must have been expanded to the same length
*/
void CpuMath::matmul(
    const float* a , Shape a_shape,
    const float* b , Shape b_shape,
    float* out, Shape out_shape
) {
    //Simple check: ¿ do they even have the max dimension ?
    if(a_shape.n_dim != TENSOR_MAX_DIM || b_shape.n_dim != TENSOR_MAX_DIM || out_shape.n_dim != TENSOR_MAX_DIM){
        throw std::runtime_error("Error: Shapes are not broadcasted");
    }

    /*Basic validation*/
    if(a_shape[TENSOR_MAX_DIM-1] != b_shape[TENSOR_MAX_DIM-2]){ throw std::runtime_error("Error: Expected shape (...,M,K); (...,K,N)  "); }
    if(a_shape[TENSOR_MAX_DIM-2] != out_shape[TENSOR_MAX_DIM-2] || b_shape[TENSOR_MAX_DIM-1] != out_shape[TENSOR_MAX_DIM-1]){ throw std::runtime_error("Error: Expected shape (...,M,N)"); }
    
    for(size_t i = TENSOR_MAX_DIM - 2 ; i-->0; ){
        if(a_shape[i] != b_shape[i] || b_shape[i] != out_shape[i] ){ throw std::runtime_error("Error: Bad broadcasting detected"); }
    }

    size_t M = out_shape[TENSOR_MAX_DIM - 2];
    size_t N = out_shape[TENSOR_MAX_DIM - 1];
    size_t K = a_shape[TENSOR_MAX_DIM - 1];
    size_t stride_am = a_shape._strides[TENSOR_MAX_DIM - 2];
    size_t stride_ak = a_shape._strides[TENSOR_MAX_DIM - 1];
    size_t stride_bk = b_shape._strides[TENSOR_MAX_DIM - 2];
    size_t stride_bn = b_shape._strides[TENSOR_MAX_DIM - 1];
    size_t stride_outm = out_shape._strides[TENSOR_MAX_DIM - 2];
    size_t stride_outn = out_shape._strides[TENSOR_MAX_DIM - 1];
    
    /*Now comes the main algorithm*/
    /* Calculate number of "batches" */
    size_t total_batches = 1;
    for(size_t i = 0; i < TENSOR_MAX_DIM - 2; ++i){
        total_batches*= a_shape[i];
    }
    
    std::vector<size_t> coords(TENSOR_MAX_DIM-2,0);
    size_t count = 0;

    while(count < total_batches){
        /*Calculate slices*/
        const float* a_slice = a;
        const float* b_slice = b;
        float* out_slice = out;
        
        for(size_t i = 0; i < TENSOR_MAX_DIM -2; ++i){
            a_slice += coords[i] * a_shape._strides[i];
            b_slice += coords[i] * b_shape._strides[i];
            out_slice += coords[i] * out_shape._strides[i];
        }

        _standard_2d_matrix_mul(
            a_slice,b_slice,out_slice,
            M,N,K,
            stride_am,stride_ak,
            stride_bk,stride_bn,
            stride_outm, stride_outn
        );
        count++;
        if(count == total_batches) break;

        /*odometer*/
        for(size_t i = TENSOR_MAX_DIM - 2; i --> 0; ){
            coords[i]++;
            if(coords[i] < a_shape[i]) break; //a_shape[i] = b_shape[i] = out_shape[i] (because of broadcasting)
            coords[i] = 0;
        }
    }

}

/*
Convolution in 2 dimensions. 
Expects:
- Float data pointer of a tensor input `in` with shape (Batch, Channel, Height, Width)
- Float data pointer of a tensor kernel `ker` with shape (N_kernels, Channel, Ker_height, Ker_width )
- Nonconst float data pointer of a tensor `out` with shape (Batch, N_kernels, Out_height, Out_width)
*/
void CpuMath::conv2D(
    const float* in, Shape in_shape,
    const float* ker, Shape ker_shape,
    float* out, Shape out_shape,
    std::array<size_t,2> stride, std::array<size_t,2> padding
) {
    //Basic validation: Dimensions
    if( in_shape.n_dim != 4 || in_shape.n_dim!=ker_shape.n_dim || in_shape.n_dim!=out_shape.n_dim ){ throw std::runtime_error("Error: Expected 4D tensors"); }


}

void CpuMath::conv3D(
    const float* in, Shape in_shape,
    const float* ker, Shape ker_shape,
    float* out, Shape out_shape,
    std::array<size_t,3> stride, std::array<size_t,3> padding
) {

}

void CpuMath::reLU(const float* a, float* out, size_t n_elements) {

}

void CpuMath::sigmoid(const float* a, float* out, size_t n_elements) {

}

void CpuMath::tanh(const float* a, float* out, size_t n_elements) {

}

void CpuMath::softmax(const float* a, float* out, size_t n_elements) {

}

void CpuMath::sum(
    const float* a, Shape a_shape, 
    float* out, Shape out_shape ,
    size_t axis
) {

}
