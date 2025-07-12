cmake_minimum_required(VERSION 3.10)

if(PROJECT_BINARY_DIR EQUAL PROJECT_SOURCE_DIR)
  message(
    WARNING "Binary directory of CMake CANNOT be the same as Source Directory!")
endif()

file(GLOB_RECURSE source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

project(
  concurrency
  VERSION 0.0.1
  LANGUAGES C CXX)

add_subdirectory(hierarchical_lock)
add_subdirectory(thread_pool)
add_subdirectory(circular_queue)
add_subdirectory(linklist)
add_subdirectory(queue)
add_subdirectory(stack)
#add_library(concurrency STATIC ${source_file} ${header_file})
#target_include_directories(concurrency PUBLIC include)
#set_property(TARGET concurrency PROPERTY POSITION_INDEPENDENT_CODE ON)
#set_target_properties(concurrency PROPERTIES LINKER_LANGUAGE CXX)
