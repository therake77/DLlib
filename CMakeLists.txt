cmake_minimum_required(VERSION 3.18)
project(llcnn VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    set(CMAKE_CUDA_ARCHITECTURES native)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(__USE_CUDA__)

else()
    message(WARNING "CUDA compiler not found. GPU support will be disabled.")
endif()


include_directories(project/includes)

set(SRC_FILES
    project/src/core/ConvLayer.cpp
    project/src/core/FullyConnected.cpp
    project/src/core/ActivationLayer.cpp
    project/src/core/BatchNormLayer.cpp
    project/src/core/Pooling.cpp
    project/src/core/Flatten.cpp
    project/src/core/OutputLayer.cpp
    project/src/core/Tensor.cpp
    project/src/core/Shape.cpp
    project/src/core/ConvNeuralNetwork.cpp
    project/src/data/stb_image.cpp
    project/src/data/ImageLoader.cpp
    project/src/data/Dataset.cpp
    project/src/hardware/CpuDevice.cpp
    project/src/hardware/DeviceManager.cpp
    project/src/math/CpuMath.cpp
    project/src/math/TensorMath.cpp
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND SRC_FILES project/src/hardware/GpuDevice.cu)
    list(APPEND SRC_FILES project/src/math/GpuMath.cu )
endif()

add_library(cnn_lib STATIC ${SRC_FILES})

target_include_directories(cnn_lib PUBLIC project/includes)

add_executable(tensor_test tests/tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE cnn_lib)

add_executable(conv_test tests/conv_test.cpp)
target_link_libraries(conv_test PRIVATE cnn_lib)

add_executable(conv_layer_test tests/conv_layer_test.cpp)
target_link_libraries(conv_layer_test PRIVATE cnn_lib)

add_executable(pooling_test tests/pooling_test.cpp)
target_link_libraries(pooling_test PRIVATE cnn_lib)

add_executable(cnn_test tests/cnn_test.cpp)
target_link_libraries(cnn_test PRIVATE cnn_lib)

add_executable(dataset_test tests/dataset_test.cpp)
target_link_libraries(dataset_test PRIVATE cnn_lib)

add_executable(dataset_real_test tests/dataset_real_test.cpp)
target_link_libraries(dataset_real_test PRIVATE cnn_lib)

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(tensor_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(conv_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(conv_layer_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(pooling_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(cnn_test PRIVATE CUDA::cuda_driver)
endif()