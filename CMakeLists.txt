cmake_minimum_required(VERSION 3.18)
if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

project(llcnn VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
    
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
    project/src/data/stb_image_resize.cpp
    project/src/data/ImageLoader.cpp
    project/src/data/Dataset.cpp
    project/src/data/DataLoader.cpp
    project/src/data/FlatDataset.cpp
    project/src/data/transforms/CenterCrop.cpp
    project/src/data/transforms/Compose.cpp
    project/src/data/transforms/Normalize.cpp
    project/src/data/transforms/Resize.cpp
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

target_include_directories(cnn_lib PUBLIC include)

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(cnn_lib PRIVATE CUDA::curand)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options( cnn_lib PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:-G>"      
            "$<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>"
        )
    endif()
    target_link_libraries(cnn_lib PUBLIC CUDA::curand CUDA::cuda_driver)
    target_link_options(cnn_lib INTERFACE
        $<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB:LIBCMT>
    )
endif()

if(MSVC)
    target_compile_options(cnn_lib PRIVATE
        "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Zi>"
        "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:/Od>"
    )
    target_link_options(cnn_lib PRIVATE
        $<$<CONFIG:Debug>:/DEBUG:FULL>
    )
endif()


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

add_executable(batch_norm_test tests/batch_norm.cpp)
target_link_libraries(batch_norm_test PRIVATE cnn_lib)

add_executable(kernel_test tests/kernel_tests.cpp)
target_link_libraries(kernel_test PRIVATE cnn_lib)

add_executable(dataset_test tests/dataset_test.cpp)
target_link_libraries(dataset_test PRIVATE cnn_lib)

add_executable(dataloader_test tests/dataloader_test.cpp)
target_link_libraries(dataloader_test PRIVATE cnn_lib)

add_executable(dataset_transform_test tests/dataset_transform_test.cpp)
target_link_libraries(dataset_transform_test PRIVATE cnn_lib)

add_executable(resize_test tests/resize_test.cpp)
target_link_libraries(resize_test PRIVATE cnn_lib)

add_executable(normalize_test tests/normalize_test.cpp)
target_link_libraries(normalize_test PRIVATE cnn_lib)

add_executable(center_crop_test tests/center_crop_test.cpp)
target_link_libraries(center_crop_test PRIVATE cnn_lib)

add_executable(dataset_real_test tests/dataset_real_test.cpp)
target_link_libraries(dataset_real_test PRIVATE cnn_lib)

add_executable(dataset_real_transform_test tests/dataset_real_transform_test.cpp)
target_link_libraries(dataset_real_transform_test PRIVATE cnn_lib)

add_executable(flat_mnist_dataset_test tests/flat_mnist_dataset_test.cpp)
target_link_libraries(flat_mnist_dataset_test PRIVATE cnn_lib)

add_executable(fc_only_test tests/fc_only_test.cpp)
target_link_libraries(fc_only_test PRIVATE cnn_lib)

add_executable(main project/app.cpp)
target_link_libraries(main PRIVATE cnn_lib)

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(tensor_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(conv_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(conv_layer_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(pooling_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(cnn_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(kernel_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(fc_only_test PRIVATE CUDA::cuda_driver)
    target_link_libraries(main PRIVATE CUDA::cuda_driver)
endif()

if(MSVC)
    
    target_compile_options(flat_mnist_dataset_test PRIVATE
        $<$<CONFIG:Debug>:/Zi>
        $<$<CONFIG:Debug>:/Od>
    )
    target_link_options(flat_mnist_dataset_test PRIVATE
        $<$<CONFIG:Debug>:/DEBUG:FULL>
    )

    target_compile_options(cnn_test PRIVATE
        $<$<CONFIG:Debug>:/Zi>
        $<$<CONFIG:Debug>:/Od>
    )
    target_link_options(cnn_test PRIVATE
        $<$<CONFIG:Debug>:/DEBUG:FULL>
    )

    target_compile_options(main PRIVATE
        $<$<CONFIG:Debug>:/Zi>
        $<$<CONFIG:Debug>:/Od>
    )
    target_link_options(main PRIVATE
        $<$<CONFIG:Debug>:/DEBUG:FULL>
    )
endif()

add_subdirectory(benchmarks)
