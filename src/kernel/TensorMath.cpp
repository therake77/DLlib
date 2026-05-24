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
}
