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
}