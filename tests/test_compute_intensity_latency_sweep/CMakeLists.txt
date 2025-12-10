cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(BENCHMARK_DOWNLOAD_DEPENDENCIES ON)

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

project(
test_compute_intensity_latency_sweep
  VERSION 0.0.1
  LANGUAGES C CXX CUDA)

file(GLOB TEST_SOURCES *.cc *.cpp)

add_executable(test_compute_intensity_latency_sweep ${TEST_SOURCES} )
target_include_directories(test_compute_intensity_latency_sweep PUBLIC include)

target_link_libraries(
test_compute_intensity_latency_sweep
  PUBLIC libTransferOverlap)
