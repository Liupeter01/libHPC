cmake_minimum_required(VERSION 3.10)

if(PROJECT_BINARY_DIR STREQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

project(
  RadixSortCPU
  VERSION 0.0.1
  LANGUAGES C CXX)

if(APPLE)
  set(HB_PREFIX "/opt/homebrew")

  if(EXISTS "${HB_PREFIX}/opt/llvm/bin/clang++")
    set(CMAKE_CXX_COMPILER
        "${HB_PREFIX}/opt/llvm/bin/clang++"
        CACHE FILEPATH "" FORCE)
    set(CMAKE_C_COMPILER
        "${HB_PREFIX}/opt/llvm/bin/clang"
        CACHE FILEPATH "" FORCE)
  endif()

  list(APPEND CMAKE_PREFIX_PATH "${HB_PREFIX}/opt/llvm"
       "${HB_PREFIX}/opt/libomp")
endif()

find_package(OpenMP REQUIRED)

add_library( RadixSortCPU STATIC ${source_file} ${header_file})
target_include_directories(  RadixSortCPU PUBLIC include)
target_link_libraries( RadixSortCPU  PRIVATE OpenMP::OpenMP_CXX)
set_property(TARGET   RadixSortCPU PROPERTY POSITION_INDEPENDENT_CODE ON)
set_target_properties(  RadixSortCPU PROPERTIES LINKER_LANGUAGE CXX)
