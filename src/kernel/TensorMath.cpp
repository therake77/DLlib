#include <kernel/TensorMath.hpp>
#include <stdexcept>

namespace {
	void ensure_same_shape(const Tensor& a, const Tensor& b) {
		if (a.size() != b.size() || a.get_shape() != b.get_shape()) {
			throw std::runtime_error("Tensor shape mismatch");
		}
	}

	void ensure_cpu_only(const Tensor& a) {
		if (a.get_device()->get_type() != DeviceType::CPU) {
			throw std::runtime_error("GPU backend not implemented yet");
		}
	}
}

namespace TensorMath {
    //Element-wise operations
	Tensor add(Tensor& a, Tensor& b) {
		ensure_same_shape(a, b);
		ensure_cpu_only(a);
		Tensor out(std::vector<size_t>(a.get_shape()), a.get_device());
		BaseMath::add(a, b, out);
		return out;
	}

	Tensor sub(Tensor& a, Tensor& b) {
		ensure_same_shape(a, b);
		ensure_cpu_only(a);
		Tensor out(std::vector<size_t>(a.get_shape()), a.get_device());
		BaseMath::sub(a, b, out);
		return out;
	}

	Tensor scale(Tensor& a, float n) {
		ensure_cpu_only(a);
		Tensor out(std::vector<size_t>(a.get_shape()), a.get_device());
		BaseMath::scale(a, n, out);
		return out;
	}

	Tensor multiply(Tensor& a, Tensor& b) {
		ensure_same_shape(a, b);
		ensure_cpu_only(a);
		Tensor out(std::vector<size_t>(a.get_shape()), a.get_device());
		BaseMath::multiply(a, b, out);
		return out;
	}

    //Activation functions
	Tensor reLU(Tensor& a) {
		ensure_cpu_only(a);
		Tensor out(std::vector<size_t>(a.get_shape()), a.get_device());
		BaseMath::reLU(a, out);
		return out;
	}

	Tensor sigmoid(Tensor& a) {
		ensure_cpu_only(a);
		Tensor out(std::vector<size_t>(a.get_shape()), a.get_device());
		BaseMath::sigmoid(a, out);
		return out;
	}

	Tensor tanh(Tensor& a) {
		ensure_cpu_only(a);
		Tensor out(std::vector<size_t>(a.get_shape()), a.get_device());
		BaseMath::tanh(a, out);
		return out;
	}

	Tensor softmax(Tensor& a) {
		ensure_cpu_only(a);
		Tensor out(std::vector<size_t>(a.get_shape()), a.get_device());
		BaseMath::softmax(a, out);
		return out;
	}

    //Linear Algebra
	Tensor matmul(Tensor& a, Tensor& b) {
		ensure_cpu_only(a);
		ensure_cpu_only(b);
		const auto& a_shape = a.get_shape();
		const auto& b_shape = b.get_shape();
		if (a_shape.size() != 2 || b_shape.size() != 2) {
			throw std::runtime_error("matmul requires 2D tensors");
		}
		if (a_shape[1] != b_shape[0]) {
			throw std::runtime_error("matmul shape mismatch");
		}
		std::vector<size_t> out_shape = { a_shape[0], b_shape[1] };
		Tensor out(out_shape, a.get_device());
		BaseMath::matmul(a, b, out);
		return out;
	}

	Tensor conv(Tensor& in, Tensor& ker, size_t stride, size_t padding) {
		ensure_cpu_only(in);
		ensure_cpu_only(ker);
		const auto& in_shape = in.get_shape();
		const auto& ker_shape = ker.get_shape();
		if (in_shape.size() != 2 || ker_shape.size() != 2) {
			throw std::runtime_error("conv requires 2D tensors");
		}
		size_t in_h = in_shape[0];
		size_t in_w = in_shape[1];
		size_t k_h = ker_shape[0];
		size_t k_w = ker_shape[1];
		size_t padded_h = in_h + 2 * padding;
		size_t padded_w = in_w + 2 * padding;
		if (padded_h < k_h || padded_w < k_w) {
			throw std::runtime_error("conv kernel larger than input");
		}
		if (((padded_h - k_h) % stride) != 0 || ((padded_w - k_w) % stride) != 0) {
			throw std::runtime_error("conv stride/padding mismatch");
		}
		size_t out_h = (padded_h - k_h) / stride + 1;
		size_t out_w = (padded_w - k_w) / stride + 1;
		std::vector<size_t> out_shape = { out_h, out_w };
		Tensor out(out_shape, in.get_device());
		BaseMath::conv(in, ker, out, stride, padding);
		return out;
	}
}
