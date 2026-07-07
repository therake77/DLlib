include(FetchContent)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.5
)

FetchContent_MakeAvailable(googlebenchmark)

add_executable(cnn_benchmark
    cnn_benchmark.cpp
)

target_link_libraries(cnn_benchmark PRIVATE
    cnn_lib
    benchmark::benchmark
    benchmark::benchmark_main
)

if(CMAKE_CUDA_COMPILER)
    target_link_libraries(cnn_benchmark PRIVATE CUDA::cuda_driver)
endif()
