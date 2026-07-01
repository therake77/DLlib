#include <math/GpuMath.cuh>
#include <stdexcept>
#include <cmath>
#ifdef __USE_CUDA__

#include <cassert>
namespace{

    struct Gpu1DKernelDimension{
        std::size_t num_blocks_per_grid;
        std::size_t num_threads_per_block;
    };

    inline Gpu1DKernelDimension _compute_1D_dispatch_dimensions(Shape a){
        std::size_t total_elements = a.n_elements;
        std::size_t blocks_per_grid = (total_elements + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK; 
        return {
            blocks_per_grid,
            THREADS_PER_BLOCK
        };
    };

    __device__ void _compute_coords_from_idx(const std::size_t idx,std::size_t* coords,const std::size_t* shapes,const std::size_t n_dims){
        std::size_t remaining = idx;
        for(std::size_t i = TENSOR_MAX_DIM; i-->0; ){
            if(i < n_dims){
                coords[i] = remaining % shapes[i];
                remaining/=shapes[i];
            }else{ 
                coords[i] = 0;
            }
        }
    }

    __device__ std::size_t _compute_idx_from_coords(std::size_t* coords,const std::size_t* strides,const std::size_t n_dims){
        std::size_t idx = 0;
        for(std::size_t i = n_dims; i --> 0;) {
            idx += coords[i] * strides[i];
        }
        return idx;
    }


    __global__ void add_kernel(
        const float* in_1,
        const std::size_t* in_1_strides,
        const float* in_2,
        const std::size_t* in_2_strides,
        float* out,
        const std::size_t* out_strides,
        const std::size_t* shapes,
        const std::size_t n_dims,
        const std::size_t n_elements
    ){
        using dim_t = std::size_t;
        dim_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_elements) return;
        dim_t coords[TENSOR_MAX_DIM];
        _compute_coords_from_idx(idx,coords,shapes,n_dims);
        dim_t in_1_idx = _compute_idx_from_coords(coords,in_1_strides,n_dims);
        dim_t in_2_idx = _compute_idx_from_coords(coords,in_2_strides,n_dims);
        dim_t out_idx = _compute_idx_from_coords(coords,out_strides,n_dims);
        out[out_idx] = in_1[in_1_idx] + in_2[in_2_idx];
    };

    __global__ void sub_kernel(
        const float* in_1,
        const std::size_t* in_1_strides,
        const float* in_2,
        const std::size_t* in_2_strides,
        float* out,
        const std::size_t* out_strides,
        const std::size_t* shapes,
        const std::size_t n_dims,
        const std::size_t n_elements
    ){
        using dim_t = std::size_t;
        dim_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_elements) return;
        dim_t coords[TENSOR_MAX_DIM];
        _compute_coords_from_idx(idx,coords,shapes,n_dims);
        dim_t in_1_idx = _compute_idx_from_coords(coords,in_1_strides,n_dims);
        dim_t in_2_idx = _compute_idx_from_coords(coords,in_2_strides,n_dims);
        dim_t out_idx = _compute_idx_from_coords(coords,out_strides,n_dims);
        out[out_idx] = in_1[in_1_idx] - in_2[in_2_idx];
    };

    __global__ void multiply_kernel(
        const float* in_1,
        const std::size_t* in_1_strides,
        const float* in_2,
        const std::size_t* in_2_strides,
        float* out,
        const std::size_t* out_strides,
        const std::size_t* shapes,
        const std::size_t n_dims,
        const std::size_t n_elements
    ){
        using dim_t = std::size_t;
        dim_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_elements) return;
        dim_t coords[TENSOR_MAX_DIM];
        _compute_coords_from_idx(idx,coords,shapes,n_dims);
        dim_t in_1_idx = _compute_idx_from_coords(coords,in_1_strides,n_dims);
        dim_t in_2_idx = _compute_idx_from_coords(coords,in_2_strides,n_dims);
        dim_t out_idx = _compute_idx_from_coords(coords,out_strides,n_dims);
        out[out_idx] = in_1[in_1_idx] * in_2[in_2_idx];
    };

    __global__ void scale_kernel(
        const float* in,
        const std::size_t* in_strides,
        const float factor,
        float* out,
        const std::size_t* out_strides,
        const std::size_t* shapes,
        const std::size_t n_dims,
        const std::size_t n_elements
    ){
        using dim_t = std::size_t;
        dim_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_elements) return;
        dim_t coords[TENSOR_MAX_DIM];
        _compute_coords_from_idx(idx,coords,shapes,n_dims);
        dim_t in_idx = _compute_idx_from_coords(coords,in_strides,n_dims);
        dim_t out_idx = _compute_idx_from_coords(coords,out_strides,n_dims);
        out[out_idx] = factor * in[in_idx] ;
    };

    __global__ void divide_kernel(
        const float* in_1,
        const std::size_t* in_1_strides,
        const float* in_2,
        const std::size_t* in_2_strides,
        float* out,
        const std::size_t* out_strides,
        const std::size_t* shapes,
        const std::size_t n_dims,
        const std::size_t n_elements
    ){
        using dim_t = std::size_t;
        dim_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_elements) return;
        dim_t coords[TENSOR_MAX_DIM];
        _compute_coords_from_idx(idx,coords,shapes,n_dims);
        dim_t in_1_idx = _compute_idx_from_coords(coords,in_1_strides,n_dims);
        dim_t in_2_idx = _compute_idx_from_coords(coords,in_2_strides,n_dims);
        dim_t out_idx = _compute_idx_from_coords(coords,out_strides,n_dims);
        out[out_idx] = in_1[in_1_idx] / in_2[in_2_idx];
    };

    __global__ void sqrt_kernel(
        const float* in,
        const std::size_t* in_strides,
        float* out,
        const std::size_t* out_strides,
        const std::size_t* shapes,
        const std::size_t n_dims,
        const std::size_t n_elements
    ){
        using dim_t = std::size_t;
        dim_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_elements) return;
        dim_t coords[TENSOR_MAX_DIM];
        _compute_coords_from_idx(idx,coords,shapes,n_dims);
        dim_t in_idx = _compute_idx_from_coords(coords,in_strides,n_dims);
        dim_t out_idx = _compute_idx_from_coords(coords,out_strides,n_dims);
        out[out_idx] = sqrt(in[in_idx]);
    };

    __global__ void pow_kernel(
        const float* in,
        const std::size_t* in_strides,
        int _n,
        float* out,
        const std::size_t* out_strides,
        const std::size_t* shapes,
        const std::size_t n_dims,
        const std::size_t n_elements
    ){
        using dim_t = std::size_t;
        dim_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_elements) return;
        dim_t coords[TENSOR_MAX_DIM];
        _compute_coords_from_idx(idx,coords,shapes,n_dims);
        dim_t in_idx = _compute_idx_from_coords(coords,in_strides,n_dims);
        dim_t out_idx = _compute_idx_from_coords(coords,out_strides,n_dims);
        out[out_idx] = pow(in[in_idx],_n);
    };

    __global__ void log_kernel(
        const float* in,
        const std::size_t* in_strides,
        float* out,
        const std::size_t* out_strides,
        const std::size_t* shapes,
        const std::size_t n_dims,
        const std::size_t n_elements
    ){
        using dim_t = std::size_t;
        dim_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_elements) return;
        dim_t coords[TENSOR_MAX_DIM];
        _compute_coords_from_idx(idx,coords,shapes,n_dims);
        dim_t in_idx = _compute_idx_from_coords(coords,in_strides,n_dims);
        dim_t out_idx = _compute_idx_from_coords(coords,out_strides,n_dims);
        out[out_idx] = log(in[in_idx]);
    };

    __global__ void reLU_kernel(
        const float* in,
        float* out,
        const std::size_t n_elements
    ){
        using dim_t = std::size_t;
        dim_t idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= n_elements) return;
        float value = in[idx];
        out[idx] = value > 0.0f? value : 0.0f;
    };

    __global__ void matmul_kernel(
        const float* a,
        const std::size_t* a_shapes,
        const std::size_t* a_strides,
        const float* b,
        const std::size_t* b_shapes,
        const std::size_t* b_strides,
        float* out,
        const std::size_t* out_shapes,
        const std::size_t* out_strides,
        std::size_t M,std::size_t N,std::size_t K,
        std::size_t n_dim
    ){
        std::size_t batch_idx = blockIdx.z;
        std::size_t batch_coords[TENSOR_MAX_DIM];
        _compute_coords_from_idx(batch_idx,batch_coords,out_shapes,n_dim);
        std::size_t a_batch_idx = _compute_idx_from_coords(batch_coords,a_strides,n_dim);  
        std::size_t b_batch_idx = _compute_idx_from_coords(batch_coords,a_strides,n_dim);  
        std::size_t out_batch_idx = _compute_idx_from_coords(batch_coords,a_strides,n_dim);  

        const float* a_slice = a + a_batch_idx;
        const float* b_slice = b + b_batch_idx;
        float* out_slice = out + out_batch_idx;

        std::size_t row = blockIdx.y * TILE_SIZE + blockIdx.y;
        std::size_t col = blockIdx.x * TILE_SIZE + blockIdx.x;

        __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
        __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
        float acc = 0.0f;

        std::size_t n_tiles = (K + TILE_SIZE - 1 )/TILE_SIZE;
        for(std::size_t i = 0; i < n_tiles; i++){
            //Load tile_A
            std::size_t k_row = i*TILE_SIZE + threadIdx.y;
            std::size_t k_col = i*TILE_SIZE + threadIdx.x;
            if(row < M && k_col < K){
                tile_A[threadIdx.y][threadIdx.x] = a_slice[
                    row * a_strides[n_dim - 2] + k_col * a_strides[n_dim-1]
                ];
            }else{
                tile_A[threadIdx.y][threadIdx.x] = 0.0f;
            }

            //Load tile_B
            if(k_row < M && col < K){
                tile_B[threadIdx.y][threadIdx.x] = b_slice[
                    k_row * b_strides[n_dim - 2] + col * b_strides[n_dim-1]
                ];
            }else{
                tile_B[threadIdx.y][threadIdx.x] = 0.0f;
            }
            __syncthreads();
            for(std::size_t k = 0; k < TILE_SIZE; k++){
                acc += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
            }
            __syncthreads();
        }
        if(row < M && col < N){
            out_slice[row*out_strides[n_dim-2] + col * out_strides[n_dim - 1]] = acc;
        }
        return;
    }

};


//Math backend API
void GpuMath::add(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {
    assert(a_shape == b_shape);
    assert(a_shape == out_shape);
    std::size_t n_dims = out_shape.n_dim;
    std::size_t n_elements = out_shape.n_elements;
    Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(a_shape);
    add_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
        a,
        a_shape._strides.data(),
        b,
        b_shape._strides.data(),
        out,
        out_shape._strides.data(),
        a_shape._shapes.data(),
        n_dims,
        n_elements
    );

    auto status_code = cudaDeviceSynchronize();
    if(status_code != cudaSuccess){
        throw std::runtime_error("Error: kernel execution failed");
    }
    return;
}

void GpuMath::sub(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {
    assert(a_shape == b_shape);
    assert(a_shape == out_shape);
    std::size_t n_dims = out_shape.n_dim;
    std::size_t n_elements = out_shape.n_elements;
    Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(a_shape);
    sub_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
        a,
        a_shape._strides.data(),
        b,
        b_shape._strides.data(),
        out,
        out_shape._strides.data(),
        a_shape._shapes.data(),
        n_dims,
        n_elements
    );

    auto status_code = cudaDeviceSynchronize();
    if(status_code != cudaSuccess){
        throw std::runtime_error("Error: kernel execution failed");
    }
    return;
}

void GpuMath::scale(const float* a, float* out, float _n, Shape shape) {

    std::size_t n_dims = shape.n_dim;
    std::size_t n_elements = shape.n_elements;
    Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(shape);
    scale_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
        a,
        shape._strides.data(),
        _n,
        out,
        shape._strides.data(),
        shape._shapes.data(),
        n_dims,
        n_elements
    );

    auto status_code = cudaDeviceSynchronize();
    if(status_code != cudaSuccess){
        throw std::runtime_error("Error: kernel execution failed");
    }
    return;
}

void GpuMath::multiply(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {
    assert(a_shape == b_shape);
    assert(a_shape == out_shape);
    std::size_t n_dims = out_shape.n_dim;
    std::size_t n_elements = out_shape.n_elements;
    Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(a_shape);
    multiply_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
        a,
        a_shape._strides.data(),
        b,
        b_shape._strides.data(),
        out,
        out_shape._strides.data(),
        a_shape._shapes.data(),
        n_dims,
        n_elements
    );

    auto status_code = cudaDeviceSynchronize();
    if(status_code != cudaSuccess){
        throw std::runtime_error("Error: kernel execution failed");
    }
    return;
}

void GpuMath::divide(const float* a,  Shape a_shape, const float* b, Shape b_shape, float* out, Shape out_shape) {
    assert(a_shape == b_shape);
    assert(a_shape == out_shape);
    std::size_t n_dims = out_shape.n_dim;
    std::size_t n_elements = out_shape.n_elements;
    Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(a_shape);
    divide_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
        a,
        a_shape._strides.data(),
        b,
        b_shape._strides.data(),
        out,
        out_shape._strides.data(),
        a_shape._shapes.data(),
        n_dims,
        n_elements
    );

    auto status_code = cudaDeviceSynchronize();
    if(status_code != cudaSuccess){
        throw std::runtime_error("Error: kernel execution failed");
    }
    return;
}

void GpuMath::sqrt(const float* a,  Shape a_shape, float* out, Shape out_shape) {
    assert(a_shape == out_shape);

    std::size_t n_dims = out_shape.n_dim;
    std::size_t n_elements = out_shape.n_elements;
    Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(out_shape);
    sqrt_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
        a,
        a_shape._strides.data(),
        out,
        out_shape._strides.data(),
        a_shape._shapes.data(),
        n_dims,
        n_elements
    );
    auto status_code = cudaDeviceSynchronize();
    if(status_code != cudaSuccess){
        throw std::runtime_error("Error: kernel execution failed");
    }
    return;
}

void GpuMath::pow(const float* a,  Shape a_shape, int _n, float* out, Shape out_shape) {
    assert(a_shape == out_shape);
    std::size_t n_dims = out_shape.n_dim;
    std::size_t n_elements = out_shape.n_elements;
    Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(out_shape);
    pow_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
        a,
        a_shape._strides.data(),
        _n,
        out,
        out_shape._strides.data(),
        a_shape._shapes.data(),
        n_dims,
        n_elements
    );
    auto status_code = cudaDeviceSynchronize();
    if(status_code != cudaSuccess){
        throw std::runtime_error("Error: kernel execution failed");
    }
    return;
}


void GpuMath::log(const float* a,  Shape a_shape, float* out, Shape out_shape){
    assert(a_shape == out_shape);

    std::size_t n_dims = out_shape.n_dim;
    std::size_t n_elements = out_shape.n_elements;
    Gpu1DKernelDimension dims = _compute_1D_dispatch_dimensions(out_shape);
    log_kernel<<<dims.num_blocks_per_grid,dims.num_threads_per_block>>>(
        a,
        a_shape._strides.data(),
        out,
        out_shape._strides.data(),
        a_shape._shapes.data(),
        n_dims,
        n_elements
    );
    auto status_code = cudaDeviceSynchronize();
    if(status_code != cudaSuccess){
        throw std::runtime_error("Error: kernel execution failed");
    }
    return;
}

//Linear Algebra
void GpuMath::matmul(
    const float* a , Shape a_shape,  
    const float* b , Shape b_shape,
    float* out, Shape out_shape
) {
    size_t n_dim = out_shape.n_dim;
    size_t M = out_shape[n_dim - 2];
    size_t N = out_shape[n_dim - 1];
    size_t K = a_shape[n_dim - 1];

    // Total batch count = product of all dims except last two
    size_t total_batches = 1;
    for(size_t i = 0; i < n_dim - 2; i++)
        total_batches *= out_shape[i];

    // 2D tile grid over (N, M), one layer per batch
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid(
        (N + TILE_SIZE - 1) / TILE_SIZE,   // blocks along N
        (M + TILE_SIZE - 1) / TILE_SIZE,   // blocks along M
        total_batches             // one z-slice per batch
    );

    matmul_kernel<<<grid, block>>>(
        a, a_shape._shapes.data(), a_shape._strides.data(),
        b, b_shape._shapes.data(), b_shape._strides.data(),
        out, out_shape._shapes.data(), out_shape._strides.data(),
        M, N, K, n_dim
    );

    auto result = cudaDeviceSynchronize();
    if(result != cudaSuccess){
        throw std::runtime_error("Error: kernel execution failed");
    }
    return;
}
    
void GpuMath::conv2D(
    const float* in, Shape in_shape,
    const float* ker, Shape ker_shape,
    float* out, Shape out_shape,
    std::array<size_t,2> stride, std::array<size_t,2> padding
) {
    throw std::runtime_error("Not implemented");
}

void GpuMath::im2col2d(
    const float* in, Shape in_shape,
    float* out, Shape out_shape,
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding
){

}

void GpuMath::col2im2d(
    const float* in, Shape in_shape,
    float* out, Shape out_shape,
    std::array<std::size_t,2> window_shape,
    std::array<std::size_t,2> strides,
    std::array<std::size_t,2> padding
){

}

//Activation functions
void GpuMath::reLU(
    const float* a, float* out, size_t n_elements
) {
    std::size_t blocks_per_grid = (n_elements + THREADS_PER_BLOCK - 1 )/THREADS_PER_BLOCK;
    reLU_kernel<<<blocks_per_grid,THREADS_PER_BLOCK>>>(
        a,
        out,
        n_elements
    );

    auto status_code = cudaDeviceSynchronize();
    if(status_code != cudaSuccess){
        throw std::runtime_error("Error: Add kernel execution failed");
    }
    return;
}

void GpuMath::sigmoid(const float* a, float* out, size_t n_elements) {

}
void GpuMath::tanh(const float* a, float* out, size_t n_elements) {

}
void GpuMath::softmax(
    const float* a,
    Shape a_shape, 
    float* out,
    Shape out_shape
) {

}
    
    //Structural ops and reshaping
void GpuMath::sum(
    const float* a, Shape a_shape, 
    float* out, Shape out_shape ,
    size_t axis
) {

}

    //Pooling operations
void GpuMath::max_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    float* mask,
    const Shape mask_shape,
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
) {

}

void GpuMath::inv_max_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    const float* mask,
    const Shape mask_shape,
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
) {

}

void GpuMath::avg_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
) {

}

void GpuMath::inv_avg_pooling2D(
    const float* input,
    const Shape input_shape,
    std::array<std::size_t,2> strides,  
    std::array<std::size_t,2> window_shapes,    //(Height , Width)
    float* out,
    const Shape out_shape,
    std::array<std::size_t, 2> padding
) {

}
    
void GpuMath::binary_positive_mask(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape
) {
    throw std::runtime_error("Not implemented");
}

void GpuMath::mean(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape,
    std::size_t axis
) {

}

void GpuMath::variance(
    const float* in,
    const Shape& in_shape,
    const float* mean,
    const Shape& mean_shape,
    float* out,
    const Shape& out_shape,
    std::size_t axis
) {

}

void GpuMath::reduce_all(
    const float* in,
    const Shape& in_shape,
    float* out,
    const Shape& out_shape,
    std::size_t axis
) {

}

void GpuMath::transpose(
    const float* in, const Shape& in_shape,
    float* out, const Shape& out_shape
) {

}


#endif //__USE_CUDA__