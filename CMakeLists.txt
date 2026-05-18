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


include_directories(includes)

set(SRC_FILES
    src/core/Layer.cpp
    src/core/ConvLayer.cpp
    src/core/Pooling.cpp
    src/core/Tensor.cpp
    src/hardware/CpuDevice.cpp
    src/hardware/DeviceManager.cpp
)

if(CMAKE_CUDA_COMPILER)
    list(APPEND SRC_FILES src/hardware/GpuDevice.cu)
endif()

add_library(cnn_lib STATIC ${SRC_FILES})

target_include_directories(cnn_lib PUBLIC include)

add_executable(tensor_test tests/tensor_test.cpp)
target_link_libraries(tensor_test PRIVATE cnn_lib)