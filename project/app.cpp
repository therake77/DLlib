#include <Llcnn.hpp>
#include <filesystem>
#include <cstddef>
#define ADD_LAYER(v, LayerType, ...) v.push_back(std::make_unique<LayerType>(__VA_ARGS__))


namespace fs = std::filesystem;


int main(){
    auto dev = DeviceManager::get_cpu_device();
    fs::path root = "C:\\Users\\USER\\Documents\\llcnn\\datasets\\mnist\\flat";
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
        DeviceManager::get_cpu_device()
    );

    /*
        Bucle de entrenamiento
    
    */
    cnn.set_random_weights();
    cnn.set_mode(LayerMode::Training);
    std::size_t epoch = 2000;
    //Negative sign because every layer's update_weight() applies param += lr*grad (ascent),
    //so this flips it into a descent step. TODO: fix the sign at the source in each update_weight().
    float lr = -0.01;
    Tensor delta = Tensor({1});
    do{
        std::cout<<"Epoch: "<<std::to_string(epoch)<<std::endl;
        std::size_t curr_index = 0;
        std::size_t batch_size = 64;
        
        do{
            std::cout<<"Current index: "<<std::to_string(curr_index)<<std::endl;
            Tensor input = dataset.load_batch_cpu(curr_index,batch_size);
            Tensor labels = dataset.labels_one_hot_cpu(curr_index,batch_size);
            /*Result contains a (N,) vector of errors*/
            auto errors = cnn.forward(input,labels);
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
            /*Do something here with the error*/
            fancy_print(*errors);
            fancy_print(delta);

            /*Now start backward pass and update*/

            cnn.backward(input,labels);
            cnn.update_weights(lr);
            curr_index+=batch_size;
        }while(curr_index + batch_size <= 1,280);
        epoch--;
    }while (epoch > 0);


    /*Now, perform inference*/
    cnn.set_mode(LayerMode::Inference); 

    


}