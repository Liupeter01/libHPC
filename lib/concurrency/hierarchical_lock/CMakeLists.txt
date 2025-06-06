cmake_minimum_required(VERSION 3.10)

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

project(
  hierarchical_lock
  VERSION 0.0.1
  LANGUAGES C CXX)

add_library(hierarchical_lock STATIC ${source_file} ${header_file})
target_include_directories(hierarchical_lock PUBLIC include)
set_property(TARGET hierarchical_lock PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(hierarchical_lock PROPERTIES LINKER_LANGUAGE CXX)
