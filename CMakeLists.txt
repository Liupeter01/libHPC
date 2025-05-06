cmake_minimum_required(VERSION 3.10)
include(FetchContent)

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
endif()

if(NOT BUILD_WITHOUT_TEST)
  enable_testing()

  file(GLOB_RECURSE gtest_source_file CONFIGURE_DEPENDS test/src/*.cpp
       test/src/*.cc test/src/*.c)
  file(GLOB_RECURSE gtest_header_file CONFIGURE_DEPENDS test/include/*.hpp
       test/include/*.h)

  add_executable(hello_test ${gtest_header_file} ${gtest_source_file})
  target_link_libraries(hello_test GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(hello_test)
endif()

FetchContent_Declare(
  TBB
  GIT_REPOSITORY https://github.com/uxlfoundation/oneTBB
  GIT_TAG v2022.1.0
  GIT_SUBMODULES_RECURSE TRUE)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(TBB googletest)

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
    CACHE INTERNAL "libHPC include dirs")
