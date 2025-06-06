cmake_minimum_required(VERSION 3.10)
project(LIBHPC_test LANGUAGES CXX C)

include(FetchContent)

if (NOT LIBHPC_BUILD_TESTING)
    return()
endif()

file(GLOB_RECURSE gtest_source_file CONFIGURE_DEPENDS src/*.cpp src/*.cc src/*.c)
file(GLOB_RECURSE gtest_header_file CONFIGURE_DEPENDS include/*.hpp include/*.h)

  FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

add_subdirectory(test_hierarchical_lock)
add_subdirectory(test_threadpool_ref)