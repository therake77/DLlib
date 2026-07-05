#include <Llcnn.hpp>
#include <filesystem>
#include <cstddef>
#define ADD_LAYER(v, LayerType, ...) v.push_back(std::make_unique<LayerType>(__VA_ARGS__))


namespace fs = std::filesystem;
fs::path root = "datasets/mnist/flat";
fs::path train_images = root / "train_images_float32.bin";
fs::path train_labels = root / "train_labels_uint8.bin";

int main(){
    auto dev = DeviceManager::get_cpu_device();
    std::vector<std::unique_ptr<Layer>> layers;
    ADD_LAYER(
        layers, Conv2D, 
        std::array<std::size_t,4>{32,1,3,3},
        dev,
        std::array<std::size_t,2>{0,0},
        std::array<std::size_t,2>{1,1}
    );
    ADD_LAYER(
        layers,BatchNorm,
        32,dev
    );
    ADD_LAYER(
        layers,ActivationLayer,
        ActivationTypes::ReLU,
        dev
    );
    ADD_LAYER(
        layers,MaxPooling,
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
        layers,BatchNorm,
        64,dev
    );
    ADD_LAYER(
        layers,ActivationLayer,
        ActivationTypes::ReLU,
        dev
    );
    ADD_LAYER(
        layers,MaxPooling,
        std::array<std::size_t, 2>{2,2},
        std::array<std::size_t, 2>{2,2},
        std::array<std::size_t, 2>{0,0},
        dev
    );
    ADD_LAYER(
        layers,Flatten,
        dev
    );
    ADD_LAYER(
        layers,FullyConnected,
        std::array<size_t, TENSOR_MAX_DIM>{1600,128},
        dev
    );
    ADD_LAYER(
        layers,ActivationLayer,
        ActivationTypes::ReLU,
        dev
    );
    ADD_LAYER(
        layers,FullyConnected,
        std::array<size_t, TENSOR_MAX_DIM>{128,10},
        dev
    );
    ADD_LAYER(
        layers,OutputLayer,
        dev,
        10
    );
    ConvolutionalNeuralNetwork cnn = ConvolutionalNeuralNetwork( std::move(layers) );

    Tensor mnist_input_simulation = Tensor(
        {5,1,28,28},
        dev
    );
    
    /*
        Carga el dataset
    */


    FlatDataset dataset = FlatDataset(
        train_images,
        train_labels,
        60000,
        1,
        28,
        28,
        10
    );

    /*
        Bucle de entrenamiento
    
    */

    std::size_t epoch = 2000;
    
    do{
        Tensor input = 
    }while (epoch > 0);



    fancy_print(*result);


}