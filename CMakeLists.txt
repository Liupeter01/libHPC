cmake_minimum_required(VERSION 3.10)
include(FetchContent)

option(LIBHPC_BUILD_TESTING "Enable libHPC building tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(WIN32) # Windows
  add_definitions(-DNOMINMAX -D_USE_MATH_DEFINES)
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

if("${CMAKE_BUILD_TYPE}" MATCHES "[Rr]elease")
  message(STATUS "This is a Release build.")
  set(TBB_TEST
      OFF
      CACHE BOOL "Disable oneTBB tests" FORCE)
  set(LIBHPC_BUILD_TESTING
      OFF
      CACHE BOOL "Disable tests generation" FORCE)
endif()

if("${CMAKE_BUILD_TYPE}" MATCHES "[Dd]ebug")
  message(STATUS "This is a Debug build.")
endif()

if(LIBHPC_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
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

add_subdirectory(lib/sparse)
add_subdirectory(lib/hpc)
add_subdirectory(lib/concurrency)
add_subdirectory(lib/mempool)

# expose all public include paths to parent scope (optional)
set(LIBHPC_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/sparse/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/hpc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib/concurrency
    CACHE INTERNAL "libHPC include dirs")
