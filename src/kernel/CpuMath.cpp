#include <kernel/TensorMath.hpp>
#include <stdexcept>
#include <cmath>
#include <kernel/CpuMath.hpp>

void CpuMath::add(const float* a, const float* b, size_t size, float* out){
    for(size_t i = 0; i < size; i++){
        out[i] = a[i] + b[i];
    }
}

void CpuMath::sub(const float* a, const float* b, size_t size, float* out){
	for(size_t i = 0; i < size; i++){
		out[i] = a[i] - b[i];
	}
}

void CpuMath::scale(const float* a, size_t size, float _n, float* out){
    for(size_t i = 0; i < size; i++){
        out[i] = _n*a[i];
    }
}

void CpuMath::multiply(const float* a, const float* b, size_t size, float* out){
    for(size_t i = 0; i < size; i++){
        out[i] = a[i] * b[i];
    }
}

void CpuMath::matmul(const float* a, std::vector<size_t> a_shape, const float* b, std::vector<size_t> b_shape, float* out){
    size_t m = a_shape[0];
    size_t k = a_shape[1];
    size_t k_b = b_shape[0];
    size_t n = b_shape[1];

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float acc = 0.0f;
            for (size_t kk = 0; kk < k; ++kk) {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
}

void CpuMath::conv(const float* in, std::vector<size_t> in_shape, const float* ker, std::vector<size_t> ker_shape, float* out, size_t stride, size_t padding){
    size_t in_h = in_shape[0];
    size_t in_w = in_shape[1];
    size_t k_h = ker_shape[0];
    size_t k_w = ker_shape[1];

    const size_t padded_h = in_h + 2 * padding;
    const size_t padded_w = in_w + 2 * padding;

    size_t out_h = (padded_h - k_h) / stride + 1;
    size_t out_w = (padded_w - k_w) / stride + 1;

    for (size_t oy = 0; oy < out_h; ++oy) {
        for (size_t ox = 0; ox < out_w; ++ox) {
            float acc = 0.0f;
            size_t in_y = oy * stride;
            size_t in_x = ox * stride;
            for (size_t ky = 0; ky < k_h; ++ky) {
                for (size_t kx = 0; kx < k_w; ++kx) {
                    int64_t iy = static_cast<int64_t>(in_y + ky) - static_cast<int64_t>(padding);
                    int64_t ix = static_cast<int64_t>(in_x + kx) - static_cast<int64_t>(padding);
                    if (iy < 0 || ix < 0 || iy >= static_cast<int64_t>(in_h) || ix >= static_cast<int64_t>(in_w)) {
                        continue;
                    }
                    acc += in[iy * in_w + ix] * ker[ky * k_w + kx];
                }
            }
            out[oy * out_w + ox] = acc;
        }
    }

}

//Activation functions
void CpuMath::reLU(const float* a, size_t size, float* out){
    for (size_t i = 0; i < size; ++i) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

void CpuMath::sigmoid(const float* a, size_t size, float* out){
    for (size_t i = 0; i < size; ++i) {
        out[i] = 1.0f / (1.0f + std::exp(-a[i]));
    }
}

void CpuMath::tanh(const float* a, size_t size, float* out){
    for (size_t i = 0; i < size; ++i) {
        out[i] = std::tanh(a[i]);
    }
}

void CpuMath::softmax(const float* a, size_t size, float* out){
    if (size == 0) {
        return;
    }

    float max_val = a[0];
    for (size_t i = 1; i < size; ++i) {
        if (a[i] > max_val) {
            max_val = a[i];
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        out[i] = std::exp(a[i] - max_val);
        sum += out[i];
    }

    if (sum == 0.0f) {
        return;
    }

    float inv_sum = 1.0f / sum;
    for (size_t i = 0; i < size; ++i) {
        out[i] *= inv_sum;
    }
}

//Structural ops and reshaping
void CpuMath::sum(const float* a, std::vector<size_t> shape, size_t axis, float* out){
    return;
}

//TODO(Do we really need a reshape at this level?)
void CpuMath::reshape(const float* a, std::vector<size_t> shape, std::vector<size_t> new_shape, float* out){
    return;
}

void CpuMath::transpose(const float* a, std::vector<size_t> shape, float* out){
    return;
}