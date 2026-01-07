cmake_minimum_required(VERSION 3.10)
include(GoogleTest)

if(NOT LIBHPC_BUILD_TESTING)
  return()
endif()

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

set(CUDA_FILE src/test_radix_global_base.cu 
                              src/test_radix_local_count.cu 
                              src/test_radix_local_offset.cu 
                             include/gpu.cuh src/gpu.cu)

project(
  test_radixsort_gpu
  VERSION 0.0.1
  LANGUAGES C CXX CUDA)

find_package(CUDA REQUIRED)

add_executable(test_radixsort_gpu ${source_file} ${header_file} ${CUDA_FILE})
target_include_directories(test_radixsort_gpu PUBLIC include ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_radixsort_gpu PRIVATE RadixSortCPU RadixSortGPU libCudaUtilities GTest::gtest)
set_property(TARGET test_radixsort_gpu PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(test_radixsort_gpu PROPERTIES LINKER_LANGUAGE CXX)
gtest_discover_tests(test_radixsort_gpu)