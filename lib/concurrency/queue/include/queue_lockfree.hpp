#pragma once
#ifndef _QUEUE_LOCKFREE_HPP
#define _QUEUE_LOCKFREE_HPP
#include <atomic>
#include <atomic_reference_node.hpp>
#include <memory>
#include <optional>
#include <type_traits>

namespace concurrency {
template <typename _Ty> class ConcurrentQueue;
}

template <typename _Ty> class concurrency::ConcurrentQueue {
  ConcurrentQueue(const ConcurrentQueue &) = delete;
  ConcurrentQueue &operator=(const ConcurrentQueue &) = delete;

public:
  ConcurrentQueue() {
    ReferenceNode<_Ty> init_node;
    init_node.node = new Node<_Ty>;
    init_node.thread_ref_counter = 1;

    m_head.store(init_node, std::memory_order_release);
    m_tail.store(init_node, std::memory_order_release);
  }
  virtual ~ConcurrentQueue() { clear(); }

public:
  void clear() {
    while (!empty()) {
      auto res = pop();
      (void)res;
    }
    delete m_tail.load().node;
  }

  const bool empty() const {
    ReferenceNode<_Ty> old_head = m_head.load();
    return ((old_head.node == m_tail.load().node) ? true : false);
  }

  /*Maybe Approximate!*/
  const std::size_t size() const { return m_size; }

  void push(_Ty &&value) { __push(std::make_unique<_Ty>(std::move(value))); }
  void push(const _Ty &value) { __push(std::make_unique<_Ty>(value)); }

  [[nodiscard]]
  std::optional<std::unique_ptr<_Ty>> pop() {
    return __pop();
  }

protected:
  void __push(std::unique_ptr<_Ty> value) {
    ReferenceNode<_Ty> new_next;
    new_next.node = new Node<_Ty>;
    new_next.thread_ref_counter = 1;

    ReferenceNode<_Ty> old_tail = m_tail.load(std::memory_order_relaxed);
    for (;;) {
      old_tail = __increase_ref_rmw(m_tail, old_tail);

      [[maybe_unused]] _Ty *old_data{nullptr};
      if (old_tail.node->data.compare_exchange_strong(old_data, value.get())) {

        ReferenceNode<_Ty> old_next{};
        if (!old_tail.node->next.compare_exchange_strong(old_next, new_next)) {
          delete new_next.node;
          new_next = old_next;
        }

        value.release(); // release resource

        update_new_tail(old_tail, new_next);
        ++m_size;
        break;
      }

      ReferenceNode<_Ty> old_next{};
      if (old_tail.node->next.compare_exchange_strong(old_next, new_next)) {
        /*new_tail*/ old_next = /*old_tail*/ new_next;
        new_next.node = new Node<_Ty>; // for next iteration!
      }
      update_new_tail(old_tail, old_next);
    }
  }

  [[nodiscard]]
  std::optional<std::unique_ptr<_Ty>> __pop() {
    ReferenceNode<_Ty> old_head = m_head.load(std::memory_order_relaxed);
    if (!old_head.node) {
      return std::nullopt;
    }

    for (;;) {
      old_head = __increase_ref_rmw(m_head, old_head);

      /*safty consideration, UB happened*/
      if (!old_head.node) {
        __release_curr_thread_ref(old_head);
        return std::nullopt;
      }

      if (old_head.node == m_tail.load().node) {
        __release_curr_thread_ref(old_head);
        return std::nullopt;
      }

      // if old_head is wrong! then old_head.node->next.load() is also wrong!
      auto next = old_head.node->next.load();
      if (m_head.compare_exchange_strong(old_head, next)) {
        _Ty *res = old_head.node->data.exchange(nullptr);
        __remove_node_from_heap(old_head); // delete node!
        --m_size;
        return std::unique_ptr<_Ty>(res);
      }

      // old_head = m_head.load();
      __release_curr_thread_ref(old_head);
    }
  }

private:
  void update_new_tail(ReferenceNode<_Ty> &old_tail,
                       const ReferenceNode<_Ty> &new_tail) {
    Node<_Ty> *backup = old_tail.node;
    while (!m_tail.compare_exchange_weak(old_tail, new_tail) &&
           backup == old_tail.node)
      ;

    if (backup == old_tail.node)
      __sync_threads_ref(old_tail);
    else
      backup->release_curr_thread_ref();
  }

  [[nodiscard]]
  static ReferenceNode<_Ty>
  __increase_ref_rmw(AtomicReferenceNode<_Ty> &main_node,
                     ReferenceNode<_Ty> &old) {
    ReferenceNode<_Ty> new_ref;
    do {
      new_ref = old;
      new_ref.thread_ref_counter += 1;
    } while (!main_node.compare_exchange_weak(old, new_ref));
    return new_ref;
  }

  static void __sync_threads_ref(ReferenceNode<_Ty> &old) {
    if (!old.node)
      return;
    const std::intptr_t any_other_threads = old.thread_ref_counter - 2;
    old.node->sync_threads_ref(any_other_threads);
  }

  static void __release_curr_thread_ref(ReferenceNode<_Ty> &old) {
    if (!old.node)
      return;
    old.node->release_curr_thread_ref();
  }

  static const bool __remove_data(ReferenceNode<_Ty> &old) {
    if (!old.node)
      return false;
    return old.node->remove_data();
  }

  static const bool __remove_node_from_heap(ReferenceNode<_Ty> &old) {
    __sync_threads_ref(old);
    return __remove_data(old);
  }

private:
  std::atomic<std::size_t> m_size;
  AtomicReferenceNode<_Ty> m_head;
  AtomicReferenceNode<_Ty> m_tail;
};

#endif // _QUEUE_LOCKFREE_HPP
