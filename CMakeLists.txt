cmake_minimum_required(VERSION 3.10)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (WIN32)        #Windows 
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

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
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

add_subdirectory(sparse)
add_subdirectory(hpc)

# expose all public include paths to parent scope (optional)
set(LIBHPC_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/sparse/include
  ${CMAKE_CURRENT_SOURCE_DIR}/hpc/include
  CACHE INTERNAL "libHPC include dirs"
)