#include <test_stall_short.h>

void benchmark_baseline() {
          __benchmark_baseline();
}

void benchmark_dim3() {
          __benchmark_dim3();
}

void benchmark_dim3_shared() {
          __benchmark_dim3_shared();
}

void benchmark_dim3_shared_solv_conflict() {
          __benchmark_dim3_shared_solv_conflict();
}

void benchmark_all(){
          __benchmark_all();
}
