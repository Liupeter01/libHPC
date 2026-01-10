cmake_minimum_required(VERSION 3.18)

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

message(INFO
        "In Order To Review The Performance, You Have to use Profiling Tools")

if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING "CUDA not supported! Radix Sort GPU Disabled!")
  return()
endif()

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

set(CUDA_FILE include/cuda_radix_sort.cuh 
                              src/cuda_radix_sort.cu
                              include/cuda_radix_sort_v1.cuh 
                              src/cuda_radix_sort_v1.cu
                              include/cuda_radix_sort_v2.cuh
                              src/cuda_radix_sort_v2.cu
                               include/cuda_radix_sort_v3.cuh
                              src/cuda_radix_sort_v3.cu

                              include/cuda_local_histogram.cuh
                              include/cuda_radix_scatter.cuh
                              include/cuda_global_reduce_from_local.cuh
)

project(
  RadixSortGPU
  VERSION 0.0.1
  LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

add_library(RadixSortGPU STATIC ${source_file} ${header_file} ${CUDA_FILE})
target_include_directories(RadixSortGPU PUBLIC include ${CUDA_INCLUDE_DIRS})
target_link_libraries(RadixSortGPU PUBLIC libCudaUtilities)
set_property(TARGET RadixSortGPU PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(RadixSortGPU PROPERTIES LINKER_LANGUAGE CXX)
