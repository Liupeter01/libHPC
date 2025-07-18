cmake_minimum_required(VERSION 3.10)
project(LIBHPC_test LANGUAGES CXX C)

include(FetchContent)

if(NOT LIBHPC_BUILD_TESTING)
  return()
endif()

if(UNIX AND NOT APPLE)
  message(STATUS "Configuring for Linux")
  add_compile_options(-fsanitize=address,leak)
  add_link_options(-fsanitize=address,leak)
elseif(APPLE)
  message(STATUS "Configuring for macOS")
  add_compile_options(-fsanitize=address)
  add_link_options(-fsanitize=address)
  # Optional: enable leak detection (partial support)
  set(ENV{ASAN_OPTIONS} "detect_leaks=1")
elseif(WIN32)
  message(WARNING "Sanitizers not yet configured for Windows")
  # ASan for MSVC or clang-cl (more limited) You may skip or use CMake presets
  # to control this
endif()

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
add_subdirectory(test_circular_queue_lk)
add_subdirectory(test_circular_queue_lockfree)
add_subdirectory(test_linklist_lk)
add_subdirectory(test_stack_lockfree)
add_subdirectory(test_queue_lockfree)
