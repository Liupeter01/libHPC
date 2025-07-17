#pragma once
#ifndef _ATOMIC_REFERENCE_NODE_HPP_
#define _ATOMIC_REFERENCE_NODE_HPP_
#include <cassert>
#include <iostream>
#include <ref_counter_packed.hpp>

namespace concurrency {

template <typename _Ty> struct ReferenceNode;

template <typename _Ty> struct AtomicReferenceNode;

template <typename _Ty> struct Node {
  Node() : data(nullptr), inner_ref_counter{} {

    ReferenceNode<_Ty> node;
    node.node = nullptr;
    node.thread_ref_counter = 0;
    next.store(node);
  }

  void sync_threads_ref(const std::intptr_t any_other_threads) {
    inner_ref_counter.sync_threads_ref(any_other_threads);
  }

  void release_curr_thread_ref() { inner_ref_counter.dec_thread_ref(); }

  bool remove_data() {
    if (inner_ref_counter.both_zero()) {
      delete this;
      return true;
    }
    return false;
  }
  AtomicReferenceNode<_Ty> next;
  std::atomic<_Ty *> data;
  ref_counter_packed inner_ref_counter;
};

/*Unsafe!!*/
template <typename _Ty> struct alignas(16) ReferenceNode {
  std::intptr_t
      thread_ref_counter; // how many threads are referencing this node?
  Node<_Ty> *node;
};
} // namespace concurrency

namespace concurrency {

/*
 * [ 63 : PTR_BITS ] ！！ thread_ref_counter
 * [  PTR_BITS - 1 : 0 ] ！！ pointer
 */
template <typename _Ty> struct alignas(16) AtomicReferenceNode {

  using packed_t = std::uintptr_t;

#if INTPTR_MAX == INT64_MAX
  // 64-bit platform
  static constexpr int PTR_BITS = 48; // safe for canonical addressing
#elif INTPTR_MAX == INT32_MAX
  // 32-bit platform
  static constexpr int PTR_BITS = 32;
#else
#error "Unsupported pointer size"
#endif

  using reference_node = ReferenceNode<_Ty>;
  using node_pointer = Node<_Ty> *;
  static constexpr packed_t PTR_MASK = (1ULL << PTR_BITS) - 1;
  static constexpr int REF_SHIFT = PTR_BITS;

  static packed_t pack(node_pointer node, std::intptr_t ref) {
    return (reinterpret_cast<packed_t>(node) & PTR_MASK) |
           (static_cast<packed_t>(ref) << REF_SHIFT);
  }

  static node_pointer extract_ptr(packed_t val) {
    return reinterpret_cast<node_pointer>(val & PTR_MASK);
  }

  static std::intptr_t extract_ref(packed_t val) {
    // Sign-extension-safe if ref never goes negative (or if you want to allow
    // negative ref)
    return static_cast<std::intptr_t>(val >> REF_SHIFT);
  }

  // Atomic load into structured ReferenceNode
  reference_node
  load(std::memory_order order = std::memory_order_acquire) const {
    packed_t val = counter.load();
    return {extract_ref(val), extract_ptr(val)};
  }

  // Atomic store from structured ReferenceNode
  void store(const reference_node &rn,
             std::memory_order order = std::memory_order_release) {
    counter.store(pack(rn.node, rn.thread_ref_counter));
  }

  // Helpers for direct reference counter manipulation
  void inc_ref(std::memory_order order = std::memory_order_acq_rel) {
    counter.fetch_add(static_cast<packed_t>(1) << REF_SHIFT);
  }

  void dec_ref(std::memory_order order = std::memory_order_acq_rel) {
    counter.fetch_sub(static_cast<packed_t>(1) << REF_SHIFT);
  }

  // Compare-and-swap with structured reference_node
  bool
  compare_exchange_weak(reference_node &expected, const reference_node &desired,
                        std::memory_order success = std::memory_order_acq_rel,
                        std::memory_order fail = std::memory_order_acquire) {
    packed_t expected_raw = pack(expected.node, expected.thread_ref_counter);
    packed_t desired_raw = pack(desired.node, desired.thread_ref_counter);
    bool ok = counter.compare_exchange_weak(expected_raw, desired_raw);
    if (!ok) {
      expected = {extract_ref(expected_raw), extract_ptr(expected_raw)};
    }
    return ok;
  }

  bool
  compare_exchange_strong(reference_node &expected,
                          const reference_node &desired,
                          std::memory_order success = std::memory_order_acq_rel,
                          std::memory_order fail = std::memory_order_acquire) {
    packed_t expected_raw = pack(expected.node, expected.thread_ref_counter);
    packed_t desired_raw = pack(desired.node, desired.thread_ref_counter);
    bool ok = counter.compare_exchange_strong(expected_raw, desired_raw);
    if (!ok) {
      expected = {extract_ref(expected_raw), extract_ptr(expected_raw)};
    }
    return ok;
  }

  std::atomic<packed_t> counter{static_cast<packed_t>(
      0)}; // init thread_ref_counter = 0  node = nullptr(0)
};
} // namespace concurrency

#endif // !_ATOMIC_REFERENCE_NODE_HPP_
