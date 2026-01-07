cmake_minimum_required(VERSION 3.18)

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

if(NOT CMAKE_CUDA_COMPILER)
  message(WARNING "CUDA not supported! Transfer Overlap Testsuite Disabled!")
  return()
endif()

message(INFO
        "In Order To Review The Performance, You Have to use Profiling Tools")

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

set(CUDA_FILE include/cuda_tut_transfer_overlap.cuh
              src/cuda_tut_transfer_overlap.cu)

project(
  libTransferOverlap
  VERSION 0.0.1
  LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

add_library(libTransferOverlap STATIC ${source_file} ${header_file}
                                      ${CUDA_FILE})
target_include_directories(libTransferOverlap PUBLIC include
                                                     ${CUDA_INCLUDE_DIRS})
target_link_libraries(libTransferOverlap PUBLIC libCudaUtilities)
set_property(TARGET libTransferOverlap PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(libTransferOverlap PROPERTIES LINKER_LANGUAGE CXX)
