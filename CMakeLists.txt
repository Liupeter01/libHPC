cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
include(CheckLanguage)

option(LIBHPC_BUILD_TESTING "Enable libHPC building tests" ON)
check_language(CUDA)

if(WIN32) # Windows
  add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
endif()

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

if(NOT MSVC)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  message(STATUS "No CMAKE_BUILD_TYPE found, Set Release build as default.")
endif()

  set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE BOOL "Disable Google Benchmark" FORCE)

set(BENCHMARK_ENABLE_GTEST_TESTS
    OFF
    CACHE BOOL "Disable Benchmark GTest" FORCE)

if("${CMAKE_BUILD_TYPE}" MATCHES "[Rr]elease")
  message(STATUS "This is a Release build.")
  set(CMAKE_BUILD_TYPE Release)
  set(TBB_TEST
      OFF
      CACHE BOOL "Disable oneTBB tests" FORCE)
  set(LIBHPC_BUILD_TESTING
      OFF
      CACHE BOOL "Disable tests generation" FORCE)
else()
  message(STATUS "This is a Debug build.")
    set(CMAKE_BUILD_TYPE Debug)
endif()

FetchContent_Declare(
  TBB
  GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB
  GIT_TAG v2022.1.0
  GIT_SUBMODULES_RECURSE TRUE)

FetchContent_MakeAvailable(TBB)

project(
  libHPC
  VERSION 0.0.1
  LANGUAGES C CXX)

  add_subdirectory(external/benchmark)
add_subdirectory(external/libmorton)
add_subdirectory(lib/sparse)
add_subdirectory(lib/hpc)
add_subdirectory(lib/concurrency)
add_subdirectory(lib/mempool)
add_subdirectory(lib/sort)
add_subdirectory(tests)

if(CMAKE_CUDA_COMPILER)
    add_subdirectory(lib/gpu)
else()
    message(WARNING "CUDA not supported!")
endif()

if(LIBHPC_BUILD_TESTING)
  enable_testing()
endif()
