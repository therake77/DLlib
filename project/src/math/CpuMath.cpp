#include <stdexcept>
#include <cmath>
#include <math/CpuMath.hpp>
#include <vector>
#include <cassert>
#include <algorithm>
#include <string>
#include <limits>

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

    bool _increment_odometer(std::vector<size_t>& coords, std::array<size_t,TENSOR_MAX_DIM> shape){
        for(size_t i = coords.size(); i --> 0 ;){
            //guard against overflow
            if(coords[i] == SIZE_MAX){ throw std::runtime_error("Error: Overflow detected"); }
            //try incrementing it
            coords[i]++;
            //If the increment don't exceed the current dimension, then just break (returning true)
            if(coords[i] < shape[i]){
                return true;
            }else{ //Else, go back the coordinate to zero
                coords[i] = 0;
            }
            //And in the next loop, the next dimension would be incremented
        }
        //If the loop doesn't break and finishes, that means all coords are zero and the odometer has "overflowed". 
        // There is no element next.
        return false;
    }

    size_t _compute_idx_from_coords(std::vector<size_t> coords, std::array<size_t,TENSOR_MAX_DIM> strides ){
        size_t offset = 0;
        for(size_t i = 0; i < coords.size();i ++){
            offset += (coords[i] * strides[i]);
        }
        return offset;
    }

};

/*
    @brief Adds two tensors
    @param a First operand
    @param a_shape Shape of first operand
    @param b Second operand
    @param b_shape Shape of second operand
    @param out Output tensor
    @param out_shape Shape of output tensor
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::add(
    const float* a,  Shape a_shape, 
    const float* b, Shape b_shape, 
    float* out, Shape out_shape
){
    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t b_idx = _compute_idx_from_coords(coords,b_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = a[a_idx] + b[b_idx]; 
    }while(_increment_odometer(coords,shapes));
}

/*
    @brief Subtracts two tensors
    @param a Pointer to first tensor operand data
    @param a_shape Shape of first operand
    @param b Pointer to second tensor operand data
    @param b_shape Shape of second operand
    @param out Pointer to output tensor data
    @param out_shape Shape of output tensor
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::sub(
    const float* a,  Shape a_shape, 
    const float* b, Shape b_shape, 
    float* out, Shape out_shape
) {
    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t b_idx = _compute_idx_from_coords(coords,b_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = a[a_idx] - b[b_idx]; 
    }while(_increment_odometer(coords,shapes));
}


/*
    @brief Scales a tensor up to a fixed float constant
    @param a Pointer to input data container 
    @param out Pointer to output data container
    @param _n Scalar
    @param shape Shared shape (out and a must have the same shape)
    
*/
void CpuMath::scale(const float* a, float* out, float _n, Shape shape) {
    for(size_t i = 0; i < shape.n_elements; i++){
        out[i] = _n*a[i];
    }
}

/*
    @brief Performs division elementwise between `a` and `b`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param b Pointer of second input data container
    @param b_shape Shape of second input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::divide(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape){
    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t b_idx = _compute_idx_from_coords(coords,b_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = a[a_idx] / b[b_idx]; 
    }while(_increment_odometer(coords,shapes));
}

/*
    @brief Performs the square root operation elementwise over `a`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::sqrt(const float* a,  Shape a_shape, float* out, Shape out_shape){
    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = std::sqrt(a[a_idx]); 
    }while(_increment_odometer(coords,shapes));
}

/*
    @brief Performs a `pow` operation with power `_n` elementwise over `a`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::pow(const float* a,  Shape a_shape, int _n, float* out, Shape out_shape){
    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = std::pow(a[a_idx],_n); 
    }while(_increment_odometer(coords,shapes));
}


void CpuMath::log(const float* a,  Shape a_shape, float* out, Shape out_shape){
    assert(a_shape == out_shape);

    std::vector<std::size_t> coords(a_shape.n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(
            coords,
            a_shape._strides
        );
        std::size_t out_idx = _compute_idx_from_coords(
            coords,
            out_shape._strides
        );
        out[out_idx] = std::log(a[a_idx]);
    }while(_increment_odometer(coords,a_shape._shapes));
}


/*
    @brief Performs multiplication elementwise between `a` and `b`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param b Pointer of second input data container
    @param b_shape Shape of second input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::multiply(
    const float* a,  Shape a_shape, 
    const float* b, Shape b_shape, 
    float* out, Shape out_shape
) {
    assert(a_shape == b_shape);
    // All shapes are equal, and have the same dimensionality
    auto n_dim = a_shape.n_dim;
    auto shapes = a_shape._shapes;
    std::vector<std::size_t> coords(n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(coords,a_shape._strides);
        std::size_t b_idx = _compute_idx_from_coords(coords,b_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);

        out[out_idx] = a[a_idx] * b[b_idx]; 
    }while(_increment_odometer(coords,shapes));
}

/*
    @brief Performs a tensor contraction on the two innermost dimensions between `a` and `b`
    @param a Pointer of first input data container
    @param a_shape Shape of first input data container
    @param b Pointer of second input data container
    @param b_shape Shape of second input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note The method is specially designed to support broadcasted tensors, using the provided shape strides
*/
void CpuMath::matmul(
    const float* a , Shape a_shape,
    const float* b , Shape b_shape,
    float* out, Shape out_shape
) {
    
    assert(a_shape.n_dim == b_shape.n_dim && b_shape.n_dim == out_shape.n_dim);
    std::size_t max_dim = a_shape.n_dim;
    //Simple check: ¿ do they even have the max dimension ?

    /*Basic validation*/
    if(a_shape[max_dim-1] != b_shape[max_dim-2]){ throw std::runtime_error("Error: Expected shape (...,M,K); (...,K,N)  "); }
    if(a_shape[max_dim-2] != out_shape[max_dim-2] || b_shape[max_dim-1] != out_shape[max_dim-1]){ throw std::runtime_error("Error: Expected shape (...,M,N)"); }
    
    for(size_t i = max_dim - 2 ; i-->0; ){
        if(a_shape[i] != b_shape[i] || b_shape[i] != out_shape[i] ){ throw std::runtime_error("Error: Bad broadcasting detected"); }
    }

    size_t M = out_shape[max_dim - 2];
    size_t N = out_shape[max_dim - 1];
    size_t K = a_shape[max_dim - 1];
    size_t stride_am = a_shape._strides[max_dim - 2];
    size_t stride_ak = a_shape._strides[max_dim - 1];
    size_t stride_bk = b_shape._strides[max_dim - 2];
    size_t stride_bn = b_shape._strides[max_dim - 1];
    size_t stride_outm = out_shape._strides[max_dim - 2];
    size_t stride_outn = out_shape._strides[max_dim - 1];
    
    /*Now comes the main algorithm*/
    /* Calculate number of "batches" */
    size_t total_batches = 1;
    for(size_t i = 0; i < max_dim - 2; ++i){
        total_batches*= a_shape[i];
    }
    
    std::vector<size_t> coords(max_dim-2,0);
    size_t count = 0;

    do{
        /*Calculate slices*/
        const float* a_slice = a;
        const float* b_slice = b;
        float* out_slice = out;
        
        for(size_t i = 0; i < max_dim -2; ++i){
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
    }while(_increment_odometer(coords,out_shape._shapes));

}

/*
    @brief Performs 2D convolution over `in` given the batch of kernels `ker`. 
    @param in Pointer to the 4D input tensor
    @param in_shape Shape of the 4D input tensor
    @param ker Pointer to the 4D kernel tensor
    @param ker_shape Shape of the 4D kernel tensor
    @param out Pointer to the output data container
    @param out_shape Shape of the output data container
    @note It was specifically designed for 4D tensors
*/
void CpuMath::conv2D(
    const float* in, const Shape in_shape,
    const float* ker, const Shape ker_shape,
    float* out, const Shape out_shape,
    std::array<size_t,2> stride, std::array<size_t,2> padding
) {
    //Basic validation:
    //Dimensions
    if( in_shape.n_dim != 4 || in_shape.n_dim!=ker_shape.n_dim || in_shape.n_dim!=out_shape.n_dim ){ throw std::runtime_error("Error: Expected 4D tensors"); }
    //Shapes
    size_t batch = in_shape[0];
    size_t channel = in_shape[1];
    size_t n_kernels = ker_shape[0];
    if( 
        batch != out_shape[0] ||
        channel != ker_shape[1] ||
        n_kernels != out_shape[1]
    ){ 
        throw std::runtime_error("Error: Shape inconsistency");
    }

    /*This is stack memory. With the assumption of no big kernels, this is tolerable*/
    size_t patch_size = ker_shape[1] * ker_shape[2] * ker_shape[3]; //C x KH x KW 
    std::vector<float> patch(patch_size,0);

    std::vector<size_t> out_coords(out_shape.n_dim,0);
    std::vector<size_t> kernel_coords(ker_shape.n_dim,0);
    std::vector<size_t> patch_coords(3,0);
    std::array<size_t,TENSOR_MAX_DIM> patch_shapes = {channel, ker_shape._shapes[2], ker_shape._shapes[3]};

    do{
        if(out_coords[1] == 0){ //For all kernels, the patches are the same. So start "gathering" when the "n_kernels" coord resets
            size_t p_idx = 0;
            //Gather algorithm
            std::fill(patch_coords.begin(),patch_coords.end(),0);
            do{
                //out_coords layout = (Batch, N_kernels, Height, Width)
                //patch_coords layout = (Batch, output_height, output_width)
                size_t in_h = out_coords[2] * stride[0] - padding[0] + patch_coords[1];
                size_t in_w = out_coords[3] * stride[1] - padding[1] + patch_coords[2];
                
                bool is_not_offset_region = in_h >= 0 && in_h < in_shape[2] && in_w >= 0 && in_w < in_shape[3];
                patch[p_idx++] = is_not_offset_region ? in[
                    _compute_idx_from_coords({out_coords[0],patch_coords[0],in_h,in_w}, in_shape._strides)
                ] : 0.0f;
            }while(_increment_odometer(patch_coords,patch_shapes));
        }
        //Compute dot product
        std::fill(kernel_coords.begin()+1,kernel_coords.end(),0);
        kernel_coords[0] = out_coords[1];
        float acc = 0.0f;
        for(size_t i = 0; i < patch_size; i++){
            acc += patch[i] * ker[_compute_idx_from_coords(
                kernel_coords, ker_shape._strides
            )];
            _increment_odometer(kernel_coords,ker_shape._shapes);
        }
        out[
            _compute_idx_from_coords(out_coords,out_shape._strides)
        ] = acc;

    }while(_increment_odometer(out_coords,out_shape._shapes));
}

/*
    @brief Given a input (image), and a series of parameters describing a 'window-sliding' operation, 
    extracts each possible window as a row of a matrix.
    @note The routine assumes `ìn` is 4-dimensional (N,C,H,W)   
*/
void CpuMath::im2col2d(
    const float* in, Shape in_shape,
    float* out, Shape out_shape,
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding
){
    using dim_t = std::size_t;
    assert(out_shape.n_dim == 2);
    //Compute the original height and width of the output (as if it was a normal window-sliding operation)
    dim_t N = in_shape[0], C = in_shape[1], H = in_shape[2], W = in_shape[3];
    dim_t H_window = window_shape[0], W_window = window_shape[1];
    
    dim_t H_out = (H + 2 * padding[0] - H_window) / strides[0] + 1;
    dim_t W_out = (W + 2 * padding[1] - W_window) / strides[1] + 1;
    //Iterate over the out array
    std::vector<std::size_t> out_coords(out_shape.n_dim,0);
    //For each index in the output matrix
    do{
        dim_t rows = out_coords[0], cols = out_coords[1];
        //Compute what element in the input tensor corresponds to this index
        dim_t delta_w = rows % W_window;
        dim_t delta_h = (rows/W_window) % H_window;
        dim_t c = rows / (H_window * W_window);
        dim_t w_out = cols % W_out;
        dim_t h_out = (cols/W_out) % H_out;
        dim_t n = cols / (H_out * W_out);
        dim_t h_in = h_out * strides[0] - padding[0] + delta_h;
        dim_t w_in = w_out * strides[1] - padding[1] + delta_w;
        dim_t out_idx = _compute_idx_from_coords(out_coords,out_shape._strides);
        if( h_in < H || w_in < W){
            dim_t in_idx = _compute_idx_from_coords({n,c,h_in,w_in}, in_shape._strides);
            out[out_idx] = in[in_idx];
        }else{
            out[out_idx] = 0.0f;
        }
    }while(_increment_odometer(out_coords,out_shape._shapes));
}

/*
    @brief Given a matrix (col), and a series of parameters describing a 'window-sliding' operation that was perfomed, 
    reconstructs the shapes of the original tensor (image), using accumulation over the overlapping indices.
    @note The routine assumes `ìn` is 2-dimensional (N,C,H,W), `out` is 4-dimensional, and out is zeroed.   
*/
void CpuMath::col2im2d(
    const float* in, Shape in_shape,
    float* out, Shape out_shape,
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding
){
    //Out should be zeroed
    //Dimensionality safeguard 
    assert(in_shape.n_dim == 2 && out_shape.n_dim == 4);
    
    using dim_t = std::size_t;
    
    //Compute the original height and width of the output (as if it was a normal window-sliding operation)
    dim_t N = out_shape[0], C = out_shape[1], H = out_shape[2], W = out_shape[3];
    
    dim_t H_window = window_shape[0], W_window = window_shape[1];
    
    dim_t H_in = (H + 2 * padding[0] - H_window) / strides[0] + 1;
    dim_t W_in = (W + 2 * padding[1] - W_window) / strides[1] + 1;
    //Iterate over the out array
    std::vector<std::size_t> in_coords(in_shape.n_dim,0);
    //For each index in the output matrix
    do{
        //Compute what element in the input tensor corresponds to this index
        dim_t rows = in_coords[0], cols = in_coords[1];
        dim_t delta_w = rows % W_window;
        dim_t delta_h = (rows/W_window) % H_window;
        dim_t c = rows / (H_window * W_window);
        dim_t w_in = cols % W_in;
        dim_t h_in = (cols/W_in) % H_in;
        dim_t n = cols / (H_in * W_in);
        dim_t h_out = h_in * strides[0] - padding[0] + delta_h;
        dim_t w_out = w_in * strides[1] - padding[1] + delta_w;
        dim_t in_idx = _compute_idx_from_coords(in_coords,in_shape._strides);
        if( h_in < H || w_in < W){
            dim_t out_idx = _compute_idx_from_coords({n,c,h_out,w_out}, out_shape._strides);
            out[out_idx] += in[in_idx];
        }
    }while(_increment_odometer(in_coords,in_shape._shapes));

}








/*
    @brief Performs reLU operation elementwise over `a`
    @param a Pointer of input data container
    @param a_shape Shape of input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note This method does not support broadcasting between `a` and `out`
    @note This method can safely accept that `a` points to the same container as `out`
*/
void CpuMath::reLU(const float* a, float* out, size_t n_elements) {
    for(size_t i = 0; i < n_elements; i++){
        out[i] = std::max(a[i],0.0f);
    }
}

/*
    @brief Performs sigmoid function operation elementwise over `a`
    @param a Pointer of input data container
    @param a_shape Shape of input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note This method does not support broadcasting between `a` and `out`
    @note This method can safely accept that `a` points to the same container as `out`
*/
void CpuMath::sigmoid(const float* a, float* out, size_t n_elements) {
    for(size_t i = 0; i < n_elements; i++){
        if(a[i] >= 0.0f){
            out[i] = 1/(1+std::exp(-a[i]));
        }else{
            float temp = std::exp(a[i]);
            out[i] = temp/(1+temp);
        }
    }
}

/*
    @brief Performs tanh function operation elementwise over `a`
    @param a Pointer of input data container
    @param a_shape Shape of input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note This method does not support broadcasting between `a` and `out`
    @note This method can safely accept that `a` points to the same container as `out`
*/
void CpuMath::tanh(const float* a, float* out, size_t n_elements) {
    for(size_t i = 0; i < n_elements; i++){
        out[i] = std::tanh(a[i]);
    }
}

/*
    @brief Performs softmax operation elementwise over `a`
    @param a Pointer of input data container
    @param a_shape Shape of input data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @note This method does not support broadcasting between `a` and `out`
    @note This method can safely accept that `a` points to the same container as `out`
    @note `a` MUST BE a 1D tensor
*/
void CpuMath::softmax(
    const float* a,
    Shape a_shape, 
    float* out,
    Shape out_shape
) {
    assert(out_shape == a_shape);
    std::vector<std::size_t> out_coords(out_shape.n_dim,0);
    std::size_t last_dim = out_shape._shapes[out_shape.n_dim-1];
    std::size_t a_last_stride = a_shape._strides[out_shape.n_dim-1];
    std::size_t out_last_stride = out_shape._strides[out_shape.n_dim-1];

    do{
        if(out_coords[out_shape.n_dim-1] == 0){
            std::size_t a_base_idx = _compute_idx_from_coords(
                out_coords,
                a_shape._strides
            );

            std::size_t out_base_idx = _compute_idx_from_coords(
                out_coords,
                out_shape._strides
            );

            std::vector<float> acc(last_dim,0);
            float sum = 0;
            float max = std::numeric_limits<float>::lowest();
            //First, find the maximum element
            for(std::size_t i = 0; i < last_dim; i++){
                std::size_t a_global_idx = a_base_idx + i * a_last_stride;
                if(a[a_global_idx] > max) max = a[a_global_idx]; 
            }
            //Now, compute e^{x - max}
            for(std::size_t i = 0; i < last_dim; i++){
                std::size_t a_global_idx = a_base_idx + i * a_last_stride;
                std::size_t out_global_idx = out_base_idx + i * out_last_stride;
                float num = std::exp(a[a_global_idx] - max);
                out[out_global_idx] = num; 
                sum +=  num;
            }
            //Divide by the accumulated sum
            for(std::size_t i = 0; i < last_dim; i++){
                std::size_t out_global_idx = out_base_idx + i * out_last_stride;
                out[out_global_idx] /= sum; 
            }
        }
    }while(_increment_odometer(out_coords,out_shape._shapes));
    return;
}

/*
    @brief Sum two tensors along a specified axis
    @note Out should be zeroes before invoking this routine
*/
void CpuMath::sum(
    const float* a, Shape a_shape, 
    float* out, Shape out_shape ,
    size_t axis
) {
    assert(a_shape == out_shape);
    std::size_t axis_dim = a_shape[axis];
    std::vector<std::size_t> coords(a_shape.n_dim,0);
    do{
        std::size_t a_idx = _compute_idx_from_coords(
            coords,
            a_shape._strides
        );
        std::size_t out_idx = _compute_idx_from_coords(
            coords,
            out_shape._strides
        );
        out[out_idx] += a[a_idx];
    }while(_increment_odometer(coords,a_shape._shapes));
}

/*
    @brief Slides a window over `input`, extracting the maximum value and storing the position of the maximum value on the original tensor as a binary mask
    @param input Pointer of input data container
    @param input_shape Shape of input data container
    @param strides The strides of the sliding window
    @param window_shapes The dimensions of the sliding window. Is assumed is a 2D window
    @param mask Pointer to the mask data container
    @param mask_shape Shape of the mask data container
    @param out Pointer of output data container
    @param out_shape Shape of output data container
    @param padding Padding for the sliding window operation
*/
void CpuMath::max_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    float* mask,
    const Shape mask_shape,
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
){
    //Both input and output should be 4 dimensional
    assert(input_shape.n_dim == 4 && out_shape.n_dim == 4);
    
    //Odometer holders
    std::vector<std::size_t> in_coords(input_shape.n_dim,0);
    std::vector<std::size_t> out_coords(out_shape.n_dim,0);
    std::vector<std::size_t> patch_coords(2,0);
    std::array<std::size_t, TENSOR_MAX_DIM> patch_shapes = {window_shapes[0],window_shapes[1]};   // Channel, Height, Width

    do{
        std::fill(patch_coords.begin(),patch_coords.end(),0);
        float max_value = std::numeric_limits<float>::lowest();
        std::size_t max_value_idx = 0;
        do{
            //Compute the coordinates of the input tensor
            std::size_t in_h = out_coords[2] * strides[0] - padding[0] + patch_coords[0];
            std::size_t in_w = out_coords[3] * strides[1] - padding[1] + patch_coords[1];
            std::size_t idx =  _compute_idx_from_coords({out_coords[0],out_coords[1], in_h, in_w},input_shape._strides);
            //If the computed coordinates are not in the padding region, compare the actual value
            if( in_h < input_shape[2] && in_h >= 0 && in_w >= 0 && in_w < input_shape[3]){
                float value = input[idx];
                mask[idx] = 0.0f;
                if( max_value < value ){
                    max_value = value;
                    max_value_idx = idx;
                }
            }else{
                //The padding region. Value is zero
                if( max_value < 0 ) max_value = 0;
            }
        }while(_increment_odometer(patch_coords,patch_shapes));
        out[ _compute_idx_from_coords(out_coords,out_shape._strides) ] = max_value;
        mask[max_value_idx] = 1.0f;
    }while(_increment_odometer(out_coords,out_shape._shapes));

    return;
}

void CpuMath::inv_max_pooling2D(
    const float* max_pooled,
    const Shape max_pooled_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    const float* mask,
    const Shape mask_shape,
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
){
    //Both input and output should be 4 dimensional
    assert( max_pooled_shape.n_dim == 4 && out_shape.n_dim == 4);
    
    //Odometer holders
    std::vector<std::size_t> pool_coords(max_pooled_shape.n_dim,0);    //
    std::vector<std::size_t> patch_coords(2,0);
    std::array<std::size_t, TENSOR_MAX_DIM> patch_shapes = {window_shapes[0],window_shapes[1]};   // Channel, Height, Width

    do{
        std::fill(patch_coords.begin(),patch_coords.end(),0);
        do{
            //Compute the coordinates of the output tensor
            std::size_t out_h = pool_coords[2] * strides[0] - padding[0] + patch_coords[0];
            std::size_t out_w = pool_coords[3] * strides[1] - padding[1] + patch_coords[1];
            std::size_t out_idx =  _compute_idx_from_coords({pool_coords[0],pool_coords[1], out_h, out_w},out_shape._strides);
            //If the computed coordinates are not in the padding region, compare the actual value
            if( out_h < out_shape[2] && out_h >= 0 && out_w >= 0 && out_w < out_shape[3]){
                if(mask[out_idx] == 1){
                    std::size_t pooled_idx = _compute_idx_from_coords(pool_coords,max_pooled_shape._strides);
                    out[out_idx] = max_pooled[pooled_idx];
                }else{
                    out[out_idx] = 0;
                }
            }
        }while(_increment_odometer(patch_coords,patch_shapes));
    }while(_increment_odometer(pool_coords,max_pooled_shape._shapes));

    return;
}


void CpuMath::avg_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
){
    //Both input and output should be 4 dimensional
    assert(input_shape.n_dim == 4 && out_shape.n_dim == 4);
    
    //Odometer holders
    std::vector<std::size_t> in_coords(input_shape.n_dim,0);
    std::vector<std::size_t> out_coords(out_shape.n_dim,0);
    std::vector<std::size_t> patch_coords(2,0);
    std::array<std::size_t, TENSOR_MAX_DIM> patch_shapes = {window_shapes[0],window_shapes[1]};   // Channel, Height, Width

    std::size_t window_n_elements = window_shapes[0] * window_shapes[1];

    do{
        std::fill(patch_coords.begin(),patch_coords.end(),0);
        float sum {0.0f};
        do{
            //Compute the coordinates of the input tensor
            std::size_t in_h = out_coords[2] * strides[0] - padding[0] + patch_coords[0];
            std::size_t in_w = out_coords[3] * strides[1] - padding[1] + patch_coords[1];
            
            //If the computed coordinates are not in the padding region, compare the actual value
            if( in_h < input_shape[2] && in_h >= 0 && in_w >= 0 && in_w < input_shape[3]){
                float value = input[ _compute_idx_from_coords(
                    {out_coords[0],out_coords[1], in_h, in_w},
                    input_shape._strides
                )];
                sum+=value;
            }
            //If is in the padding region, do nothing (sum + 0 = sum)
        }while(_increment_odometer(patch_coords,patch_shapes));
        out[ _compute_idx_from_coords(out_coords,out_shape._strides) ] = sum/window_n_elements;
    }while(_increment_odometer(out_coords,out_shape._shapes));

    return;
}

void CpuMath::inv_avg_pooling2D(
    const float* avg_pooled,
    const Shape avg_pooled_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
){
    //Just for safety: put zeros on out. It is crucially important
    std::vector<std::size_t> out_coords(out_shape.n_dim,0);
    do{
        out[_compute_idx_from_coords(out_coords,out_shape._strides)] = 0;
    }while(_increment_odometer(out_coords,out_shape._shapes));
    //Now starts the logic
    std::vector<std::size_t> pooled_coords(avg_pooled_shape.n_dim,0);
    std::vector<std::size_t> patch_coords(2,0);
    std::size_t window_n_elements = window_shapes[0] * window_shapes[1];
    do{
        std::size_t pooled_idx = _compute_idx_from_coords(pooled_coords,avg_pooled_shape._strides);
        float to_add = avg_pooled[pooled_idx]/window_n_elements;
        std::fill(patch_coords.begin(),patch_coords.end(),0);
        do{
            std::size_t out_h = pooled_coords[2] * strides[0] - padding[0] + patch_coords[0];
            std::size_t out_w = pooled_coords[3] * strides[1] - padding[1] + patch_coords[1];
            std::size_t out_idx =  _compute_idx_from_coords({pooled_coords[0],pooled_coords[1], out_h, out_w},out_shape._strides);
            if( out_h < out_shape[2] && out_h >= 0 && out_w >= 0 && out_w < out_shape[3]){
                out[out_idx] += to_add;
            }
        }while(_increment_odometer(patch_coords,out_shape._shapes));
    }while(_increment_odometer(pooled_coords,avg_pooled_shape._shapes));
    return;
}

void CpuMath::_transpose2D(
    const float* in, const Shape& in_shape,
    float* out, const Shape& out_shape
){
    size_t rows = in_shape[0];
    size_t cols = in_shape[1];

    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){

            size_t src =
                i * in_shape._strides[0] +
                j * in_shape._strides[1];

            size_t dst =
                j * out_shape._strides[0] +
                i * out_shape._strides[1];

            out[dst] = in[src];
        }
    }
}

void CpuMath::transpose(
    const float* in, const Shape& in_shape,
    float* out, const Shape& out_shape
) {
    if(in_shape.n_dim != out_shape.n_dim){
        throw std::runtime_error("Error: transpose requires matching tensor ranks");
    }
    if(in_shape.n_dim < 2){
        std::memcpy(out, in, in_shape.n_elements * sizeof(float));
        return;
    }
    if(in_shape.n_dim == 2){
        _transpose2D(in, in_shape, out, out_shape);
        return;
    }

    std::array<size_t, TENSOR_MAX_DIM> expected_out = in_shape._shapes;

    size_t last = in_shape.n_dim - 1;
    size_t prev = in_shape.n_dim - 2;

    size_t tmp = expected_out[prev];
    expected_out[prev] = expected_out[last];
    expected_out[last] = tmp;

    if(expected_out != out_shape._shapes){
        throw std::runtime_error("Error: transpose output shape mismatch");
    }

    auto index_from_coords = [](const Shape& shape, const std::array<size_t, TENSOR_MAX_DIM>& c){
        size_t idx = 0;
        for(size_t i = 0; i < shape.n_dim; ++i){
            idx += c[i] * shape._strides[i];
        }
        return idx;
    };

    std::array<size_t, TENSOR_MAX_DIM> coords{};
    coords.fill(0);

    for(size_t processed = 0; processed < out_shape.n_elements; ++processed){
        size_t src_idx = 0;

        for(size_t i = 0; i < in_shape.n_dim; ++i){
            size_t coord;
            if(i == prev){
                coord = coords[last];
            } else if(i == last){
                coord = coords[prev];
            } else{
                coord = coords[i];
            }
            src_idx += coord * in_shape._strides[i];
        }

        size_t dst_idx = index_from_coords(out_shape, coords);
        out[dst_idx] = in[src_idx];

        for(size_t dim = out_shape.n_dim; dim > 0; --dim){
            size_t i = dim - 1;
            coords[i]++;
            if(coords[i] < out_shape[i]){
                break;
            }
            coords[i] = 0;
        }
    }
}
/*
    @brief Filters positive values from `in`, zeroing negative values. Shapes must be equal
*/
void CpuMath::binary_positive_mask(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape
){
    //Iterate both tensors using a unique odometer
    std::vector<std::size_t> coords(out_shape.n_dim,0);
    do{
        std::size_t in_idx = _compute_idx_from_coords(coords,in_shape._strides);
        std::size_t out_idx = _compute_idx_from_coords(coords,out_shape._strides);
        out[out_idx] = in[in_idx] > 0 ? 1 : 0;
    }while(_increment_odometer(coords,out_shape._shapes));
}

void CpuMath::mean(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape,
    std::size_t axis
){
    //It will be assumed that out_shape is 1 dimensional 
    assert(axis < in_shape.n_dim);
    std::size_t acc_dim = in_shape[axis];
    assert(out_shape.n_dim == 1 && out_shape[0] == acc_dim);

    //As a safety measure, put zeros on out
    for(std::size_t i = 0; i < out_shape[0]; i++){
        out[i] = 0.0f;
    }

    //Prepare the odometer
    std::vector<std::size_t> coords(in_shape.n_dim,0);
    do{
        out[coords[axis]] += in[_compute_idx_from_coords(coords,in_shape._strides)];
    }while(_increment_odometer(coords,in_shape._shapes));
    //The total number of elements is the product of all shapes. So is just shape.n_elements/acc_dim
    float M = in_shape.n_elements/acc_dim;
    for(std::size_t i = 0 ; i < acc_dim; i++){
        out[i]/=M;
    }
    return;
}

void CpuMath::variance(
    const float* in,
    const Shape& in_shape,
    const float* mean,
    const Shape& mean_shape,
    float* out,
    const Shape& out_shape,
    std::size_t axis
){
    //It will be assumed that out and mean are 1-dimensional
    assert(axis < in_shape.n_dim);
    std::size_t acc_dim = in_shape[axis];

    assert(out_shape.n_dim == 1 && mean_shape.n_dim == 1);
    assert(out_shape[0] == acc_dim && mean_shape[0] == acc_dim);

    //As a safety measure, put zeros on out
    for(std::size_t i = 0; i < out_shape[0]; i++){
        out[i] = 0.0f;
    }    

    //Prepare the odometer
    std::vector<std::size_t> coords(in_shape.n_dim,0);
    do{
        std::size_t curr_acc_idx = coords[axis]; 
        std::size_t curr_idx = _compute_idx_from_coords(coords,in_shape._strides); 
        out[curr_acc_idx] += std::pow(in[curr_idx] - mean[curr_acc_idx],2);
    }while(_increment_odometer(coords,in_shape._shapes));
    //The total number of elements is the product of all shapes. So is just shape.n_elements/in[acc_dim]
    float M = in_shape.n_elements/acc_dim;
    for(std::size_t i = 0 ; i < acc_dim; i++){
        out[i]/=M;
    }
    return;
}

void CpuMath::reduce_all(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape,
    std::size_t axis
){
    //out_shape MUST BE 1 dimensional (this is a reduction of all axis except one)
    assert(axis < in_shape.n_dim);
    std::size_t acc_dim = in_shape[axis];
    assert(out_shape.n_dim == 1 && out_shape[0] == acc_dim);

    //As a safety measure, put zeros on out
    for(std::size_t i = 0; i < out_shape[0]; i++){
        out[i] = 0.0f;
    }

    //Prepare the odometer
    std::vector<std::size_t> coords(in_shape.n_dim,0);
    do{
        out[coords[axis]] += in[_compute_idx_from_coords(coords,in_shape._strides)];
    }while(_increment_odometer(coords,in_shape._shapes));
    //End: out now has the reduced sums
    return;
}