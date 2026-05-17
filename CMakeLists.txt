cmake_minimum_required(VERSION 3.10.0)
project(llcnn VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

include_directories(includes)

add_library(cnn_lib STATIC
    includes/cnn.hpp
    src/cnn.cpp
    includes/cnn/tensor.hpp
    src/cnn/tensor.cpp
    includes/cnn/backends/gpu_backend.cu
    src/cnn/backends/gpu_backend.cpp
    includes/cnn/backends/cpu_backend.hpp
    src/cnn/backends/cpu_backend.cpp
    includes/cnn/layers/base_layer.hpp
    src/cnn/layers/base_layer.cpp
    includes/cnn/layers/cnn_layer.hpp
    src/cnn/layers/cnn_layer.cpp
    includes/cnn/layers/pooling.hpp
    src/cnn/layers/pooling.cpp
)

