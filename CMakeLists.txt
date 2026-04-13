cmake_minimum_required(VERSION 4.0)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.9/bin/nvcc)
project(WattServe CXX CUDA)

find_package(CUDAToolkit 12.9 REQUIRED)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES "80")

# set(NVBENCH_DIR ${PROJECT_SOURCE_DIR}/3rdparty/nvbench
#     CACHE PATH "Path to nvbench directory")
# message(STATUS "NVBENCH dir is ${NVBENCH_DIR}")

set(CUTLASS_DIR ${PROJECT_SOURCE_DIR}/3rdparty/cutlass
    CACHE PATH "Path to cutlass directory")
message(STATUS "CUTLASS dir is ${CUTLASS_DIR}")

set(FLASHINFER_DIR ${PROJECT_SOURCE_DIR}/3rdparty/flashinfer
    CACHE PATH "Path to flashinfer directory")
message(STATUS "FLASHINFER dir is ${FLASHINFER_DIR}")



# set(FMT_DIR ${PROJECT_SOURCE_DIR}/3rdparty/fmt
#     CACHE PATH "Path to fmt directory")
# message(STATUS "FMT dir is ${FMT_DIR}")


# add_subdirectory(${NVBENCH_DIR})
# add_subdirectory(${FMT_DIR})
set(CUTLASS_INCLUDE_DIR ${CUTLASS_DIR}/include ${CUTLASS_DIR}/tools/util/include)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CUTLASS_INCLUDE_DIR})
include_directories(${FLASHINFER_DIR}/include)
# include_directories(${NVBENCH_DIR})
# include_directories(${FMT_DIR}/include)

link_directories(${NVSHMEM_DIR}/lib)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr -diag-suppress 2908,3012,3013 -Xptxas -v)
list(APPEND CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr -diag-suppress 2908,3012,3013 -Xptxas -v")
message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")


############################## OLD TARGETS BELOW ##############################

# add_executable(DistGEMM tests/dist_gemm.cu)
# set_target_properties(DistGEMM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# # target_compile_options(DistGEMM PRIVATE ${CMAKE_CUDA_FLAGS})
# target_link_libraries(DistGEMM fmt::fmt-header-only)
# target_link_libraries(DistGEMM MPI::MPI_CXX)
# target_link_libraries(DistGEMM nvshmem)
# target_link_libraries(DistGEMM nccl cublasLt cublas)

# add_executable(NVSHMEM tests/nvshmem.cu)
# set_target_properties(NVSHMEM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# # target_compile_options(NVSHMEM PRIVATE ${CMAKE_CUDA_FLAGS})
# target_link_libraries(NVSHMEM fmt::fmt-header-only)
# target_link_libraries(NVSHMEM MPI::MPI_CXX)
# target_link_libraries(NVSHMEM nvshmem)
# target_link_libraries(NVSHMEM nccl cublasLt cublas)

# add_executable(Norm tests/norm.cu)
# set_target_properties(Norm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


# add_executable(CUTE tests/gemm-multi-stage.cu)
# set_target_properties(CUTE PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(CUTE PRIVATE ${CUDA_LIBRARIES} cublasLt cublas)

# add_executable(CollectiveLaunch tests/collective_launch.cu)
# set_target_properties(CollectiveLaunch PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# # target_compile_options(CollectiveLaunch PRIVATE ${CMAKE_CUDA_FLAGS})
# target_link_libraries(CollectiveLaunch MPI::MPI_CXX)
# target_link_libraries(CollectiveLaunch nvshmem)

# add_executable(NCCLGEMM tests/nccl_gemm.cu)
# set_target_properties(NCCLGEMM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# # target_compile_options(NCCLGEMM PRIVATE ${CMAKE_CUDA_FLAGS})
# target_link_libraries(NCCLGEMM MPI::MPI_CXX)
# target_link_libraries(NCCLGEMM nccl)

# add_executable(BenchGEMM tests/bench_gemm.cu)
# set_target_properties(BenchGEMM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# # target_compile_options(BenchGEMM PRIVATE ${CMAKE_CUDA_FLAGS})
# target_link_libraries(BenchGEMM MPI::MPI_CXX)
# target_link_libraries(BenchGEMM nvshmem)

# add_executable(OrderTest tests/order_test.cu)
# set_target_properties(OrderTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(SimpleGEMM tests/simple_gemm.cu)
# set_target_properties(SimpleGEMM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# # target_compile_options(SimpleGEMM PRIVATE ${CMAKE_CUDA_FLAGS})
# target_link_libraries(SimpleGEMM ${CUDA_LIBRARIES} cublasLt cublas)
# target_link_libraries(SimpleGEMM nvshmem)


############################## NEW TARGETS BELOW ##############################


# set(SGTUNNEL_AOT_KERNELS_DIR ${PROJECT_SOURCE_DIR}/csrc/aot_kernels)
# add_library(sgtunnel_gemm STATIC ${SGTUNNEL_AOT_KERNELS_DIR}/gemm.cu)
# set_target_properties(sgtunnel_gemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(TestGEMM tests/test_gemm.cu)
# set_target_properties(TestGEMM PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(TestGEMM PRIVATE sgtunnel_gemm)
# target_link_libraries(TestGEMM PRIVATE ${CUDA_LIBRARIES} cublasLt cublas)
# target_include_directories(TestGEMM PRIVATE ${PROJECT_SOURCE_DIR}/csrc)

# add_executable(TestProducer tests/test_producer.cu)
# set_target_properties(TestProducer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# # target_link_libraries(TestGEMM PRIVATE sgtunnel_gemm)
# # target_include_directories(TestGEMM PRIVATE ${PROJECT_SOURCE_DIR}/csrc)

# add_executable(TestConsumer tests/test_consumer.cu)
# set_target_properties(TestConsumer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(TestQueue tests/test_queue.cu)
# set_target_properties(TestQueue PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(TestProducerQueue tests/test_producer_queue.cu)
# set_target_properties(TestProducerQueue PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add_executable(TestComm tests/test_comm.cu)
# set_target_properties(TestComm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(TestComm PRIVATE MPI::MPI_CXX)
# target_link_libraries(TestComm PRIVATE nvshmem)

# add_executable(TestProducerConsumer tests/test_producer_consumer.cu)
# set_target_properties(TestProducerConsumer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(TestProducerConsumer PRIVATE MPI::MPI_CXX)
# target_link_libraries(TestProducerConsumer PRIVATE nvshmem)

# add_executable(TestBaseline tests/test_baseline.cu)
# set_target_properties(TestBaseline PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(TestBaseline PRIVATE MPI::MPI_CXX)
# target_link_libraries(TestBaseline PRIVATE nvshmem)

# add_executable(TestNorm tests/test_norm.cu)
# set_target_properties(TestNorm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(TestNorm PRIVATE cuda)

# add_executable(TestBandwidth tests/test_bandwidth.cu)
# set_target_properties(TestBandwidth PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(TestBandwidth PRIVATE cuda)

# add_executable(TestBandwidthNvlink tests/test_bandwidth_nvlink.cu)
# set_target_properties(TestBandwidthNvlink PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# target_link_libraries(TestBandwidthNvlink PRIVATE cuda)


# set(Python_ROOT_DIR ".venv")
# find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)
# message(STATUS "Python executable: ${Python_EXECUTABLE}")
# message(STATUS "Python Site Lib: ${Python_SITELIB}")

# set(pybind11_DIR "${Python_SITELIB}/pybind11/share/cmake/pybind11")
# find_package(pybind11 CONFIG REQUIRED)

# set(Torch_DIR "${Python_SITELIB}/torch/share/cmake/Torch")
# find_package(Torch REQUIRED)

# pybind11_add_module(sgtunnel_cpp csrc/python_api.cpp)
# target_include_directories(sgtunnel_cpp PRIVATE ${TORCH_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})
# target_include_directories(sgtunnel_cpp PRIVATE ${SGTUNNEL_AOT_KERNELS_DIR})
# target_link_libraries(sgtunnel_cpp PRIVATE ${TORCH_LIBRARIES} torch_python cuda)
