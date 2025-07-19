#pragma once
#ifndef _REF_COUNTER_PACKED_HPP_
#define _REF_COUNTER_PACKED_HPP_
#include <atomic>
#include <iostream>

namespace concurrency {
/*
 * [ 63 : 2 ] ！！ thread_ref_counter (62 or 30)
 * [  1 : 0 ] ！！ head_and_tail_ref_counter
 */
struct alignas(16) ref_counter_packed {

  using packed_t = std::intptr_t;
  static constexpr int HEAD_TAIL_BITS = 2;
  static constexpr packed_t HEAD_TAIL_MASK = (1 << HEAD_TAIL_BITS) - 1;
  static constexpr int THREAD_REF_SHIFT = HEAD_TAIL_BITS;

  bool both_zero() const { return !counter.load(std::memory_order_acquire); }

  int get_threads_ref() const {
    return static_cast<int>(counter.load(std::memory_order_acquire) >>
                            THREAD_REF_SHIFT);
  }

  int get_head_tail_ref() const {
    return static_cast<int>(counter.load(std::memory_order_acquire) &
                            HEAD_TAIL_MASK);
  }

  void inc_thread_ref() {
    packed_t old_val = counter.load(std::memory_order_relaxed);
    packed_t new_val;
    do {
      std::intptr_t thread_ref =
          static_cast<std::intptr_t>(old_val >> THREAD_REF_SHIFT);
      std::intptr_t head_tail =
          static_cast<std::intptr_t>(old_val & HEAD_TAIL_MASK);

      thread_ref += 1;

      new_val = (thread_ref << THREAD_REF_SHIFT) | (head_tail & HEAD_TAIL_MASK);
    } while (!counter.compare_exchange_weak(old_val, new_val,
                                            std::memory_order_release,
                                            std::memory_order_acquire));
  }

  void dec_thread_ref() {
    packed_t old_val = counter.load(std::memory_order_relaxed);
    packed_t new_val;
    do {
      std::intptr_t thread_ref =
          static_cast<std::intptr_t>(old_val >> THREAD_REF_SHIFT);
      std::intptr_t head_tail =
          static_cast<std::intptr_t>(old_val & HEAD_TAIL_MASK);

      thread_ref -= 1;

      new_val = (thread_ref << THREAD_REF_SHIFT) | (head_tail & HEAD_TAIL_MASK);
    } while (!counter.compare_exchange_weak(old_val, new_val,
                                            std::memory_order_release,
                                            std::memory_order_acquire));
  }

  void dec_head_tail_ref() { counter.fetch_sub(1, std::memory_order_acq_rel); }

  void sync_threads_ref(std::intptr_t any_other_threads) {
    packed_t old_val = counter.load(std::memory_order_relaxed);
    packed_t new_val;
    do {
      std::intptr_t thread_ref =
          static_cast<std::intptr_t>(old_val >> THREAD_REF_SHIFT);
      std::intptr_t head_tail =
          static_cast<std::intptr_t>(old_val & HEAD_TAIL_MASK);

      thread_ref += any_other_threads;
      head_tail -= 1;

      if (head_tail < 0) {
        throw std::runtime_error("head_tail underflow");
      }

      new_val = (thread_ref << THREAD_REF_SHIFT) | (head_tail & HEAD_TAIL_MASK);
    } while (!counter.compare_exchange_weak(old_val, new_val,
                                            std::memory_order_release,
                                            std::memory_order_acquire));
  }

  std::atomic<packed_t> counter{
      static_cast<packed_t>(0b10)}; // init head_tail = 2, thread_ref = 0
};
} // namespace concurrency

#endif // !_REF_COUNTER_PACKED_HPP_
