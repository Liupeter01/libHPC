cmake_minimum_required(VERSION 3.10)

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

project(
  queue
  VERSION 1.0.0
  LANGUAGES C CXX)

add_library(queue STATIC ${source_file} ${header_file})
target_include_directories(queue PUBLIC include)
set_property(TARGET queue PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(queue PROPERTIES LINKER_LANGUAGE CXX)
