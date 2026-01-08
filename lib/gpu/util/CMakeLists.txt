cmake_minimum_required(VERSION 3.18)

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

set(CUDA_HEADER include/cudaHelper.cuh include/linearUtils.cuh
                src/linearUtils.cu include/cudaUtils.cuh)

find_package(CUDA REQUIRED)

project(
  libCudaUtilities
  VERSION 0.0.1
  LANGUAGES C CXX CUDA)

add_library(libCudaUtilities STATIC ${source_file} ${header_file}
                                    ${CUDA_HEADER})
target_include_directories(libCudaUtilities PUBLIC include ${CUDA_INCLUDE_DIRS})
target_link_libraries(libCudaUtilities PUBLIC CUDA::cudart)
set_property(TARGET libCudaUtilities PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(libCudaUtilities PROPERTIES LINKER_LANGUAGE CXX)
