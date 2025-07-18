#pragma once
#ifndef _STACK_LOCKFREE_REF_HPP
#define _STACK_LOCKFREE_REF_HPP
#include <atomic>
#include <memory>
#include <optional>

namespace concurrency {
template <typename _Ty> class ConcurrentStackRef;
} // namespace concurrency

template <typename _Ty> class concurrency::ConcurrentStackRef {
  struct ReferenceNode;
  struct Node {
    Node(const _Ty &value)
        : data(std::make_shared<_Ty>(value)), decrease_reference(0) {}
    Node(_Ty &&value)
        : data(std::make_shared<_Ty>(std::move(value))), decrease_reference(0) {
    }
    ReferenceNode next;
    std::shared_ptr<_Ty> data;
    std::atomic<std::intptr_t> decrease_reference;
  };
  struct ReferenceNode {
    ReferenceNode() : increase_reference(0), node(nullptr) {}
    ReferenceNode(const _Ty &value)
        : increase_reference(1), node(new Node(value)) {}
    ReferenceNode(_Ty &&value)
        : increase_reference(1), node(new Node(std::move(value))) {}
    std::intptr_t increase_reference;
    Node *node;
  };

  ConcurrentStackRef(const ConcurrentStackRef &) = delete;
  ConcurrentStackRef &operator=(const ConcurrentStackRef &) = delete;

public:
  ConcurrentStackRef() : m_head(ReferenceNode{}) {}
  ~ConcurrentStackRef() { clear(); }

public:
  void clear() {
    while (!empty()) {
      auto res = pop();
      (void)res;
    }
  }
  void push(const _Ty &value) { __push(ReferenceNode(value)); }
  void push(_Ty &&value) { __push(ReferenceNode(std::move(value))); }
  std::optional<std::shared_ptr<_Ty>> pop() { return __pop(); }
  const std::size_t size() const {
    return m_size.load(std::memory_order_acquire);
  }
  const bool empty() const {
    return !m_head.load(std::memory_order_acquire).node;
  }

protected:
  void __push(const ReferenceNode &node) {
    ReferenceNode expected = m_head.load(std::memory_order_acquire);
    do {
      node.node->next = expected;
    } while (!m_head.compare_exchange_weak(
        expected, node, std::memory_order_release, std::memory_order_relaxed));

    m_size++;
  }

  std::optional<std::shared_ptr<_Ty>> __pop() {
    ReferenceNode old_head = m_head.load();

    for (;;) {
      /*New Thead Coming In And Update reference counter for current old_head
       * node!*/
      ReferenceNode new_ref = __increase_ref_rmw(old_head);

      Node *handle = new_ref.node;
      if (handle == nullptr) {
        return std::nullopt;
      }

      if (m_head.compare_exchange_strong(new_ref, handle->next,
                                         std::memory_order_relaxed)) {
        std::shared_ptr<_Ty> res = handle->data;
        --m_size;
        std::intptr_t increase = new_ref.increase_reference - 2;
        if (!(handle->decrease_reference.fetch_add(increase,
                                                   std::memory_order_release) +
              increase))
          delete handle;
        return res;
      }

      /*so, do you find out that we actuallt not using new_ref anymore!*/
      if (1 ==
          handle->decrease_reference.fetch_sub(1, std::memory_order_relaxed)) {
        (void)handle->decrease_reference.load(std::memory_order_acquire);
        delete handle;
      }
    }
  }

private:
  [[nodiscard]]
  ReferenceNode __increase_ref_rmw(ReferenceNode &old_head) {
    ReferenceNode new_ref;
    do {
      new_ref = old_head;
      new_ref.increase_reference += 1;
    } while (!m_head.compare_exchange_weak(old_head, new_ref,
                                           std::memory_order_acquire,
                                           std::memory_order_relaxed));
    return new_ref;
  }

private:
  std::atomic<std::size_t> m_size;
  std::atomic<ReferenceNode> m_head;
};

#endif
