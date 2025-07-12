#include <hierarchical_lock.hpp>

thread_local std::size_t hierarchical_lock::thread_level =
    std::numeric_limits<std::size_t>::max();

// This source file is only a placeholder to ensure the hpc_array static library
// is built.
void __ensure_concurrency_compiles() {}
