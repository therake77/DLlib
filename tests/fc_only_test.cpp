#include <Llcnn.hpp>
#include <filesystem>
#include <cstddef>
#include <limits>

#define ADD_LAYER(v, LayerType, ...) v.push_back(std::make_unique<LayerType>(__VA_ARGS__))

namespace fs = std::filesystem;

/*
    FC-only isolation test: Flatten -> FC -> ReLU -> FC -> OutputLayer.
    No Conv2D, no Pooling, no BatchNorm — the simplest possible classifier, used to check
    whether the GPU training path is numerically correct for the fully-connected + output
    stack alone. Input is a MNIST batch (N,1,28,28); Flatten gives 1*28*28 = 784 features
    -> FC(784,128) -> ReLU -> FC(128,10) -> OutputLayer(softmax + cross-entropy).

    It trains a single fixed batch of 100 images for 500 epochs at lr -0.1 (the negative
    sign flips each layer's ascent step into gradient descent). With so few samples and many
    epochs this should overfit and drive the loss down close to zero, then predict the
    training batch correctly at inference time.
*/
int main(){
    auto dev = DeviceManager::get_gpu_device();
    auto cpu_dev = DeviceManager::get_cpu_device();
    fs::path root = "C:\\Users\\USER\\Documents\\llcnn\\datasets\\mnist\\flat";
    std::cout << root << std::endl;
    fs::path train_images = root / "train_images_float32.bin";
    fs::path train_labels = root / "train_labels_uint8.bin";

    std::vector<std::unique_ptr<Layer>> layers;
    ADD_LAYER(
        layers, Flatten,
        dev
    );
    ADD_LAYER(
        layers, FullyConnected,
        std::array<size_t, TENSOR_MAX_DIM>{784,128},
        dev
    );
    ADD_LAYER(
        layers, ActivationLayer,
        ActivationTypes::ReLU,
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
    //Negative sign because every layer's update_weight() applies param += lr*grad (ascent),
    //so this flips it into a descent step. Same convention currently used in project/app.cpp.
    float lr = -0.1f;
    Tensor delta = Tensor({1},dev);
    std::size_t curr_index = 0;
    std::size_t batch_size = 100;
    std::size_t n_batchs_per_epoch = 1;

    //Load the single fixed training batch once and keep reusing it every epoch.
    Tensor input = dataset.load_batch_cpu(curr_index,batch_size);
    Tensor labels = dataset.labels_one_hot_cpu(curr_index,batch_size);
    input.to(dev);
    labels.to(dev);

    do{
        std::cout << "Epoch: " << std::to_string(epoch) << std::endl;

        std::size_t counter = 0;
        do{
            delta.to(dev);
            /*Result contains a (N,) vector of errors*/
            auto errors = cnn.forward(input,labels);
            TensorMath::sum(
                *errors,
                0,
                delta
            );
            TensorMath::scale(
                delta,
                1.0f/(float)batch_size,
                delta
            );
            delta.to(cpu_dev);
            std::cout << "Loss: " << delta.data()[0] << std::endl;

            cnn.backward(input,labels);
            cnn.update_weights(lr);
            counter++;
        }while(counter < n_batchs_per_epoch);
        epoch--;
    }while (epoch > 0);

    /*Now, perform inference over the trained batch and report accuracy.*/
    cnn.set_mode(LayerMode::Inference);

    std::size_t correct = 0;
    std::size_t total = 0;
    curr_index = 0;
    do{
        Tensor sample = dataset.load_cpu(curr_index);
        Tensor label = dataset.labels_one_hot_cpu(curr_index,1);
        sample.to(dev);
        label.to(cpu_dev);
        auto _result = cnn.forward(sample);
        Tensor result = Tensor(*_result,cpu_dev);

        std::size_t predicted = 0;
        float max_value = std::numeric_limits<float>::lowest();
        const float* data = result.data();
        for(std::size_t i = 0; i < 10; i++){
            if(data[i] > max_value){
                max_value = data[i];
                predicted = i;
            }
        }

        std::size_t true_num = 0;
        max_value = std::numeric_limits<float>::lowest();
        data = label.data();
        for(std::size_t i = 0; i < 10; i++){
            if(data[i] > max_value){
                max_value = data[i];
                true_num = i;
            }
        }

        std::cout << "Result: " << std::to_string(predicted)
                  << " | True value: " << std::to_string(true_num) << std::endl;
        if(predicted == true_num) correct++;
        total++;

        curr_index++;
    }while(curr_index < batch_size);

    std::cout << "Train-batch accuracy: " << correct << "/" << total << std::endl;

    return 0;
}
