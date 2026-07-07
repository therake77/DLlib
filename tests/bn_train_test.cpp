#include <Llcnn.hpp>
#include <filesystem>
#include <cstddef>
#include <limits>

#define ADD_LAYER(v, LayerType, ...) v.push_back(std::make_unique<LayerType>(__VA_ARGS__))

namespace fs = std::filesystem;

/*
    BatchNorm variant of fc_only_test — mirrors project/app.cpp's ORIGINAL architecture (the one
    that suffered "dead training"): Conv2D -> BatchNorm -> LeakyReLU -> MaxPool, twice, then
    Flatten -> FC -> LeakyReLU -> FC -> OutputLayer, with BatchNorm placed *before* the activation
    exactly as app.cpp originally wired it.

    Purpose: now that the GPU col2im backward bug is fixed (col2im2d wasn't zeroing its output, so
    a Conv2D feeding another layer got corrupted gradients), re-run the full BatchNorm architecture
    on the same overfit batch (100 images, 500 epochs) to determine whether the original dead
    training was caused by that gradient bug or by BatchNorm itself. Same harness/hyperparameters as
    fc_only_test (batch 100, 500 epochs, lr -0.1).

    Conv2D(1->32,3x3) 28->26 -> MaxPool 26->13 -> Conv2D(32->64,3x3) 13->11 -> MaxPool 11->5;
    Flatten -> 64*5*5 = 1600 -> FC(1600,128) -> FC(128,10).
*/
int main(){
    auto dev = DeviceManager::get_gpu_device();
    auto cpu_dev = DeviceManager::get_cpu_device();
    fs::path root = "C:\\Users\\USER\\Documents\\llcnn\\datasets\\mnist\\flat";
    fs::path train_images = root / "train_images_float32.bin";
    fs::path train_labels = root / "train_labels_uint8.bin";

    std::vector<std::unique_ptr<Layer>> layers;
    ADD_LAYER(
        layers, Conv2D,
        std::array<std::size_t,4>{32,1,3,3},
        dev,
        std::array<std::size_t,2>{0,0},
        std::array<std::size_t,2>{1,1}
    );
    ADD_LAYER(
        layers, BatchNorm,
        32, dev
    );
    ADD_LAYER(
        layers, ActivationLayer,
        ActivationTypes::LeakyReLU,
        dev
    );
    ADD_LAYER(
        layers, MaxPooling,
        std::array<std::size_t, 2>{2,2},
        std::array<std::size_t, 2>{2,2},
        std::array<std::size_t, 2>{0,0},
        dev
    );
    ADD_LAYER(
        layers, Conv2D,
        std::array<std::size_t,4>{64,32,3,3},
        dev,
        std::array<std::size_t,2>{0,0},
        std::array<std::size_t,2>{1,1}
    );
    ADD_LAYER(
        layers, BatchNorm,
        64, dev
    );
    ADD_LAYER(
        layers, ActivationLayer,
        ActivationTypes::LeakyReLU,
        dev
    );
    ADD_LAYER(
        layers, MaxPooling,
        std::array<std::size_t, 2>{2,2},
        std::array<std::size_t, 2>{2,2},
        std::array<std::size_t, 2>{0,0},
        dev
    );
    ADD_LAYER(
        layers, Flatten,
        dev
    );
    ADD_LAYER(
        layers, FullyConnected,
        std::array<size_t, TENSOR_MAX_DIM>{1600,128},
        dev
    );
    ADD_LAYER(
        layers, ActivationLayer,
        ActivationTypes::LeakyReLU,
        dev
    );
    ADD_LAYER(
        layers, FullyConnected,
        std::array<size_t, TENSOR_MAX_DIM>{128,10},
        dev
    );
    ADD_LAYER(
        layers, OutputLayer,
        dev,
        10
    );
    ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork( std::move(layers) );

    FlatDataset dataset (
        train_images,
        train_labels,
        60000,
        1,
        28,
        28,
        10,
        DeviceManager::get_cpu_device()
    );

    cnn.set_random_weights();
    cnn.set_mode(LayerMode::Training);

    std::size_t epoch = 500;
    float lr = -0.02f;
    Tensor delta = Tensor({1},dev);
    std::size_t batch_size = 100;

    Tensor input = dataset.load_batch_cpu(0,batch_size);
    Tensor labels = dataset.labels_one_hot_cpu(0,batch_size);
    input.to(dev);
    labels.to(dev);

    do{
        delta.to(dev);
        auto errors = cnn.forward(input,labels);
        TensorMath::sum(*errors, 0, delta);
        TensorMath::scale(delta, 1.0f/(float)batch_size, delta);
        delta.to(cpu_dev);
        std::cout << "Epoch " << epoch << " Loss: " << delta.data()[0] << std::endl;

        cnn.backward(input,labels);
        cnn.update_weights(lr);
        epoch--;
    }while (epoch > 0);

    cnn.set_mode(LayerMode::Inference);
    std::size_t correct = 0;
    for(std::size_t i = 0; i < batch_size; i++){
        Tensor sample = dataset.load_cpu(i);
        Tensor label = dataset.labels_one_hot_cpu(i,1);
        sample.to(dev);
        label.to(cpu_dev);
        auto _result = cnn.forward(sample);
        Tensor result = Tensor(*_result,cpu_dev);

        std::size_t predicted = 0;
        float max_value = std::numeric_limits<float>::lowest();
        const float* data = result.data();
        for(std::size_t k = 0; k < 10; k++){
            if(data[k] > max_value){ max_value = data[k]; predicted = k; }
        }
        std::size_t true_num = 0;
        max_value = std::numeric_limits<float>::lowest();
        data = label.data();
        for(std::size_t k = 0; k < 10; k++){
            if(data[k] > max_value){ max_value = data[k]; true_num = k; }
        }
        if(predicted == true_num) correct++;
    }
    std::cout << "Train-batch accuracy: " << correct << "/" << batch_size << std::endl;
    return 0;
}
