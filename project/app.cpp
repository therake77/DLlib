#include <Llcnn.hpp>
#include <filesystem>
#include <cstddef>
#include <string>
#define ADD_LAYER(v, LayerType, ...) v.push_back(std::make_unique<LayerType>(__VA_ARGS__))


namespace fs = std::filesystem;


int main(int argc, char** argv){
    bool use_gpu = false;
    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "--gpu" || mode == "gpu") {
            use_gpu = true;
        } else if (mode == "--cpu" || mode == "cpu") {
            use_gpu = false;
        } else {
            std::cerr << "Uso: " << argv[0] << " [--cpu|--gpu]" << std::endl;
            return 1;
        }
    }

    auto dev = use_gpu ? DeviceManager::get_gpu_device() : DeviceManager::get_cpu_device();
    std::cout << "Device: " << (use_gpu ? "GPU" : "CPU") << std::endl;
    fs::path root = "datasets/mnist/flat";
    std::cout << root<< std::endl;
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
    
    /*
        Carga el dataset
    */


    FlatDataset dataset (
        train_images,
        train_labels,
        60000,
        1,
        28,
        28,
        10,
        dev
    );

    /*
        Bucle de entrenamiento
    
    */
    cnn.set_random_weights();
    cnn.set_mode(LayerMode::Training);
    std::size_t epoch = 10;
    //Negative sign because every layer's update_weight() applies param += lr*grad (ascent),
    //so this flips it into a descent step. TODO: fix the sign at the source in each update_weight().
    float lr = -0.01;
    Tensor delta = Tensor({1}, dev, 0.0f, true);
    do{
        std::cout<<"Epoch: "<<std::to_string(epoch)<<std::endl;
        std::size_t curr_index = 0;
        std::size_t batch_size = 64;
        std::size_t samples_per_epoch = 970;
        
        do{
            std::cout<<"Current index: "<<std::to_string(curr_index)<<std::endl;
            Tensor input = dataset.load_batch(curr_index,batch_size);
            Tensor labels = dataset.labels_one_hot(curr_index,batch_size);
            /*Result contains a (N,) vector of errors*/
            auto errors = cnn.forward(input,labels);
            delta.zeros();
            TensorMath::sum(
                *errors,
                0,
                delta
            );
            TensorMath::scale(
                delta,
                1.0/(float)batch_size,
                delta
            );

            /*Now delta has the error*/
            Tensor delta_cpu(delta, DeviceManager::get_cpu_device());
            std::cout<<"Loss: "<<delta_cpu.data()[0]<<std::endl;

            /*Now start backward pass and update*/

            cnn.backward(input,labels);
            cnn.update_weights(lr);
            curr_index+=batch_size;
        }while(curr_index + batch_size <= samples_per_epoch);
        epoch--;
    }while (epoch > 0);


    /*Now, perform inference*/
    cnn.set_mode(LayerMode::Inference); 

    


}
