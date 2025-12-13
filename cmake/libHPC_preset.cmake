# ============================================
# libHPC Build Preset Cross-platform TBB, LLVM, CUDA, ccache guard
# ============================================

cmake_minimum_required(VERSION 3.15)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
include(CheckLanguage)

option(LIBHPC_BUILD_TESTING "Enable libHPC building tests" ON)

check_language(CUDA)

# --------------------------------------------
# Common
# --------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(NOT MSVC)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    message(STATUS "CCache enabled: ${CCACHE_PROGRAM}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_PROGRAM})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_PROGRAM})
  endif()
endif()

# --------------------------------------------
# Platform Adjustments
# --------------------------------------------

if(WIN32)
  add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
elseif(APPLE)
  message(
    FATAL_ERROR
      "Apple macOS ARM (Apple Silicon) is not supported for HPC builds.\n"
      "Reason: oneTBB and libc++ ABI compatibility issues on macOS 14/15.\n"
      "Please use Linux or Windows for full functionality.")
  return()
endif()

# ------------------------------------------------------------------
# Disable heavy bench/test libs
# ------------------------------------------------------------------
set(BENCHMARK_ENABLE_TESTING
    OFF
    CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_TESTING
    OFF
    CACHE BOOL "" FORCE)
set(BUILD_GMOCK
    OFF
    CACHE BOOL "" FORCE)

set(TBB_TEST
    OFF
    CACHE BOOL "" FORCE)
set(TBB_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(TBB_EXAMPLES
    OFF
    CACHE BOOL "" FORCE)
set(TBB_BENCH
    OFF
    CACHE BOOL "" FORCE)
set(TBB_BENCHMARK
    OFF
    CACHE BOOL "" FORCE)

# --------------------------------------------
# oneTBB Split mode
# --------------------------------------------
FetchContent_Declare(
  TBB
  GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB
  GIT_TAG v2022.1.0
  GIT_SUBMODULES_RECURSE TRUE)

FetchContent_MakeAvailable(TBB)

# --------------------------------------------
# CUDA
# --------------------------------------------
if(CMAKE_CUDA_COMPILER)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Only compile targeted arch (optional: extend here)
  set(CMAKE_CUDA_ARCHITECTURES 75 86 89 90)

else()
  message(WARNING "CUDA not detected, GPU module skipped.")
endif()
