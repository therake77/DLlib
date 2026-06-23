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
        std::runtime_error("Error: Shape inconsistency");
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

void CpuMath::conv3D(
    const float* in, Shape in_shape,
    const float* ker, Shape ker_shape,
    float* out, Shape out_shape,
    std::array<size_t,3> stride, std::array<size_t,3> padding
) {
    throw std::runtime_error("Error: not implemented");
}

void CpuMath::reLU(const float* a, float* out, size_t n_elements) {
    for(size_t i = 0; i < n_elements; i++){
        out[i] = std::max(a[i],0.0f);
    }
}

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

void CpuMath::tanh(const float* a, float* out, size_t n_elements) {
    for(size_t i = 0; i < n_elements; i++){
        out[i] = std::tanh(a[i]);
    }
}

/*Assumes the input is a 1D tensor (a vector)*/
void CpuMath::softmax(const float* a, float* out, size_t n_elements) {
    float acc = 0.0f;
    for(size_t i = 0; i < n_elements; i++){
        acc += std::exp(a[i]);
    }
    if( std::abs(acc) < std::numeric_limits<float>::epsilon() ){ throw std::runtime_error("Error: Encountered division by zero"); }
    for(size_t i = 0; i < n_elements; i++){
        out[i] = std::exp(a[i]) / acc;
    }
}

void CpuMath::sum(
    const float* a, Shape a_shape, 
    float* out, Shape out_shape ,
    size_t axis
) {
    throw std::runtime_error("Error: not implemented");
}

void CpuMath::max_pooling2D(
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

    do{
        std::fill(patch_coords.begin(),patch_coords.end(),0);
        float max_value = std::numeric_limits<float>::lowest();
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
                if( max_value < value ) max_value = value;
            }else{
                //The padding region. Value is zero
                if( max_value < 0 ) max_value = 0;
            }
        }while(_increment_odometer(patch_coords,patch_shapes));
        out[ _compute_idx_from_coords(out_coords,out_shape._strides) ] = max_value;
    }while(_increment_odometer(out_coords,out_shape._shapes));

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