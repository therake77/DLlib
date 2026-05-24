#include <kernel/TensorMath.hpp>
#include <stdexcept>
#include <cmath>

namespace {
	void validate_cpu_pair(const Tensor& a, const Tensor& b, const Tensor& out) {
		if (a.get_device()->get_type() != DeviceType::CPU ||
			b.get_device()->get_type() != DeviceType::CPU ||
			out.get_device()->get_type() != DeviceType::CPU) {
			throw std::runtime_error("CpuMath requires tensors on CPU");
		}
		if (a.size() != b.size() || a.size() != out.size()) {
			throw std::runtime_error("Tensor size mismatch");
		}
		if (a.get_shape() != b.get_shape() || a.get_shape() != out.get_shape()) {
			throw std::runtime_error("Tensor shape mismatch");
		}
	}

	void validate_cpu_single(const Tensor& a, const Tensor& out) {
		if (a.get_device()->get_type() != DeviceType::CPU ||
			out.get_device()->get_type() != DeviceType::CPU) {
			throw std::runtime_error("CpuMath requires tensors on CPU");
		}
		if (a.size() != out.size()) {
			throw std::runtime_error("Tensor size mismatch");
		}
		if (a.get_shape() != out.get_shape()) {
			throw std::runtime_error("Tensor shape mismatch");
		}
	}
}

namespace BaseMath {
    //Element-wise operations
	void add(Tensor& a, Tensor& b, Tensor& out) {
		validate_cpu_pair(a, b, out);
		const float* a_data = a.data();
		const float* b_data = b.data();
		float* out_data = out.data();
		for (size_t i = 0; i < out.size(); ++i) {
			out_data[i] = a_data[i] + b_data[i];
		}
	}

	void sub(Tensor& a, Tensor& b, Tensor& out) {
		validate_cpu_pair(a, b, out);
		const float* a_data = a.data();
		const float* b_data = b.data();
		float* out_data = out.data();
		for (size_t i = 0; i < out.size(); ++i) {
			out_data[i] = a_data[i] - b_data[i];
		}
	}

	void scale(Tensor& a, float n, Tensor& out) {
		validate_cpu_single(a, out);
		const float* a_data = a.data();
		float* out_data = out.data();
		for (size_t i = 0; i < out.size(); ++i) {
			out_data[i] = a_data[i] * n;
		}
	}

	void multiply(Tensor& a, Tensor& b, Tensor& out) {
		validate_cpu_pair(a, b, out);
		const float* a_data = a.data();
		const float* b_data = b.data();
		float* out_data = out.data();
		for (size_t i = 0; i < out.size(); ++i) {
			out_data[i] = a_data[i] * b_data[i];
		}
	}

    //Activation functions
	void reLU(Tensor& a, Tensor& out) {
		validate_cpu_single(a, out);
		const float* a_data = a.data();
		float* out_data = out.data();
		for (size_t i = 0; i < out.size(); ++i) {
			out_data[i] = a_data[i] > 0.0f ? a_data[i] : 0.0f;
		}
	}

	void sigmoid(Tensor& a, Tensor& out) {
		validate_cpu_single(a, out);
		const float* a_data = a.data();
		float* out_data = out.data();
		for (size_t i = 0; i < out.size(); ++i) {
			out_data[i] = 1.0f / (1.0f + std::exp(-a_data[i]));
		}
	}

	void tanh(Tensor& a, Tensor& out) {
		validate_cpu_single(a, out);
		const float* a_data = a.data();
		float* out_data = out.data();
		for (size_t i = 0; i < out.size(); ++i) {
			out_data[i] = std::tanh(a_data[i]);
		}
	}

	void softmax(Tensor& a, Tensor& out) {
		validate_cpu_single(a, out);
		const float* a_data = a.data();
		float* out_data = out.data();
		if (out.size() == 0) {
			return;
		}

		float max_val = a_data[0];
		for (size_t i = 1; i < out.size(); ++i) {
			if (a_data[i] > max_val) {
				max_val = a_data[i];
			}
		}

		float sum = 0.0f;
		for (size_t i = 0; i < out.size(); ++i) {
			out_data[i] = std::exp(a_data[i] - max_val);
			sum += out_data[i];
		}

		if (sum == 0.0f) {
			return;
		}

		float inv_sum = 1.0f / sum;
		for (size_t i = 0; i < out.size(); ++i) {
			out_data[i] *= inv_sum;
		}
	}

    //Linear Algebra
	void matmul(Tensor& a, Tensor& b, Tensor& out) {
		validate_cpu_pair(a, b, out);
		const auto& a_shape = a.get_shape();
		const auto& b_shape = b.get_shape();
		const auto& out_shape = out.get_shape();
		if (a_shape.size() != 2 || b_shape.size() != 2 || out_shape.size() != 2) {
			throw std::runtime_error("matmul requires 2D tensors");
		}
		size_t m = a_shape[0];
		size_t k = a_shape[1];
		size_t k_b = b_shape[0];
		size_t n = b_shape[1];
		if (k != k_b || out_shape[0] != m || out_shape[1] != n) {
			throw std::runtime_error("matmul shape mismatch");
		}

		const float* a_data = a.data();
		const float* b_data = b.data();
		float* out_data = out.data();

		for (size_t i = 0; i < m; ++i) {
			for (size_t j = 0; j < n; ++j) {
				float acc = 0.0f;
				for (size_t kk = 0; kk < k; ++kk) {
					acc += a_data[i * k + kk] * b_data[kk * n + j];
				}
				out_data[i * n + j] = acc;
			}
		}
	}

	void conv(Tensor& in, Tensor& ker, Tensor& out, size_t stride, size_t padding) {
		validate_cpu_pair(in, ker, out);
		const auto& in_shape = in.get_shape();
		const auto& ker_shape = ker.get_shape();
		const auto& out_shape = out.get_shape();
		if (in_shape.size() != 2 || ker_shape.size() != 2 || out_shape.size() != 2) {
			throw std::runtime_error("conv requires 2D tensors");
		}
		size_t in_h = in_shape[0];
		size_t in_w = in_shape[1];
		size_t k_h = ker_shape[0];
		size_t k_w = ker_shape[1];
		size_t out_h = out_shape[0];
		size_t out_w = out_shape[1];

		const size_t padded_h = in_h + 2 * padding;
		const size_t padded_w = in_w + 2 * padding;
		if (padded_h < k_h || padded_w < k_w) {
			throw std::runtime_error("conv kernel larger than input");
		}
		if (((padded_h - k_h) % stride) != 0 || ((padded_w - k_w) % stride) != 0) {
			throw std::runtime_error("conv stride/padding mismatch");
		}
		if (out_h != (padded_h - k_h) / stride + 1 || out_w != (padded_w - k_w) / stride + 1) {
			throw std::runtime_error("conv output shape mismatch");
		}

		const float* in_data = in.data();
		const float* k_data = ker.data();
		float* out_data = out.data();

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
						acc += in_data[iy * in_w + ix] * k_data[ky * k_w + kx];
					}
				}
				out_data[oy * out_w + ox] = acc;
			}
		}
	}
}