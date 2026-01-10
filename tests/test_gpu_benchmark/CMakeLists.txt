cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

project(
  test_gpu_benchmark
  VERSION 0.0.1
  LANGUAGES C CXX)

find_package(OpenMP REQUIRED)

file(GLOB TEST_SOURCES *.cc *.cpp)

add_executable(  test_gpu_benchmark ${TEST_SOURCES})
target_include_directories(
    test_gpu_benchmark PUBLIC include ${LIBHPC_INCLUDE_DIRS}
                            ${CMAKE_CURRENT_SOURCE_DIR}/external/sse2neon)

target_link_libraries(
  test_gpu_benchmark
  PUBLIC benchmark::benchmark OpenMP::OpenMP_CXX libmorton
         #hpc_array sparse_data_structure
         RadixSortGPU)
