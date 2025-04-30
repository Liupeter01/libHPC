cmake_minimum_required(VERSION 3.10)

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
    message(WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

project(
  HighDimensionFlatArray
  VERSION 0.0.1
  LANGUAGES C CXX)

add_library(hpc_array OBJECT ${source_file} ${header_file})
set_property(TARGET hpc_array PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(hpc_array PROPERTIES LINKER_LANGUAGE CXX)