#include <Llcnn.hpp>

#include <cstddef>
#include <filesystem>
#include <limits>

#define ADD_LAYER(v, LayerType, ...) v.push_back(std::make_unique<LayerType>(__VA_ARGS__))

namespace fs = std::filesystem;

int main() {
    auto dev = DeviceManager::get_gpu_device();
    auto cpu_dev = DeviceManager::get_cpu_device();

    fs::path root = "datasets/mnist/flat";
    std::cout << root << std::endl;
    fs::path train_images = root / "train_images_float32.bin";
    fs::path train_labels = root / "train_labels_uint8.bin";

    std::vector<std::unique_ptr<Layer>> layers;
    ADD_LAYER(layers, Flatten, dev);
    ADD_LAYER(
        layers,
        FullyConnected,
        std::array<std::size_t, TENSOR_MAX_DIM>{784, 128},
        dev
    );
    ADD_LAYER(layers, ActivationLayer, ActivationTypes::ReLU, dev);
    ADD_LAYER(
        layers,
        FullyConnected,
        std::array<std::size_t, TENSOR_MAX_DIM>{128, 10},
        dev
    );
    ADD_LAYER(layers, OutputLayer, dev, 10);

    ConvolutionalNeuralNetwork cnn(std::move(layers));

    FlatDataset dataset(
        train_images,
        train_labels,
        60000,
        1,
        28,
        28,
        10,
        dev
    );

    cnn.set_random_weights();
    cnn.set_mode(LayerMode::Training);

    std::size_t epoch = 200;
    std::size_t batch_size = 100;
    std::size_t curr_index = 0;
    float lr = -0.01f;
    Tensor delta({1}, dev, 0.0f, true);

    while(epoch > 0) {
        Tensor input = dataset.load_batch(curr_index, batch_size);
        Tensor labels = dataset.labels_one_hot(curr_index, batch_size);

        auto errors = cnn.forward(input, labels);
        delta.zeros();
        TensorMath::sum(*errors, 0, delta);
        TensorMath::scale(delta, 1.0f / static_cast<float>(batch_size), delta);

        Tensor delta_cpu(delta, cpu_dev);
        std::cout << "Epoch: " << epoch << " Loss: " << delta_cpu.data()[0] << std::endl;

        cnn.backward(input, labels);
        cnn.update_weights(lr);
        --epoch;
    }

    cnn.set_mode(LayerMode::Inference);

    for(std::size_t i = 0; i <= 20; ++i) {
        Tensor input = dataset.load(i);
        Tensor label = dataset.labels_one_hot(i, 1);
        Tensor label_cpu(label, cpu_dev);

        auto output = cnn.forward(input);
        Tensor output_cpu(*output, cpu_dev);

        std::size_t predicted = 0;
        float max_value = std::numeric_limits<float>::lowest();
        for(std::size_t c = 0; c < 10; ++c) {
            if(output_cpu.data()[c] > max_value) {
                max_value = output_cpu.data()[c];
                predicted = c;
            }
        }

        std::size_t expected = 0;
        max_value = std::numeric_limits<float>::lowest();
        for(std::size_t c = 0; c < 10; ++c) {
            if(label_cpu.data()[c] > max_value) {
                max_value = label_cpu.data()[c];
                expected = c;
            }
        }

        std::cout << "Result: " << predicted << " True value: " << expected << std::endl;
    }

    return 0;
}
