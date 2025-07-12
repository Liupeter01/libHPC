#pragma once
#ifndef _STACK_LOCKFREE_HPP
#define _STACK_LOCKFREE_HPP
#include <atomic>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>

namespace concurrency {
template <typename _Ty> class ConcurrentStack;

namespace details {

#define MAX_HAZARD_NUMBER 128

class hazard_manager;

struct hazard_pointer {
  hazard_pointer() { clear(); }
  ~hazard_pointer() { clear(); }

  void clear() {
    id = std::thread::id{};
    pointer = nullptr;
  }

  std::atomic<std::thread::id> id{std::thread::id{}};
  std::atomic<void *> pointer{nullptr};
};

class hazard_data {

  friend class hazard_manager;

  template <typename _Ty>
  using node_pointer = typename concurrency::ConcurrentStack<_Ty>::Node *;

  /*reclaim list*/
  template <typename _Ty> struct ReclaimNode {
    ReclaimNode(node_pointer<_Ty> m) : next(nullptr), node(m) {}

    ~ReclaimNode() {
      if (node)
        delete node;
      next = nullptr;
    }

    node_pointer<_Ty> node{nullptr};
    ReclaimNode *next{nullptr};
  };

  /*hazard array*/
  static hazard_pointer hazard_pointers[MAX_HAZARD_NUMBER];

  /*reclaim linklist*/
  template <typename _Ty> using reclaim_pointer = ReclaimNode<_Ty> *;

  template <typename _Ty> static std::atomic<reclaim_pointer<_Ty>> reclaim_head;
};

class hazard_manager {
  hazard_manager(const hazard_manager &) = delete;
  hazard_manager &operator=(const hazard_manager &) = delete;

public:
  hazard_manager() : hp(nullptr) {
    hp = find_unoccupancy();
    if (!hp)
      throw std::runtime_error("Unexpected Error Occured!");
  }

  ~hazard_manager() {
    if (!hp)
      return;
    hp->id.store(std::thread::id{});
    hp->pointer.store(nullptr);
  }

public:
  [[nodiscard]] std::atomic<void *> &get_pointer() {
    if (!hp)
      hp = find_unoccupancy();
    return hp->pointer;
  }

  static bool check_reference(const void *p) {
    if (!p)
      return false;

    for (auto &h : hazard_data::hazard_pointers)
      if (p == h.pointer.load())
        return true;
    return false;
  }

  template <typename _Ty>
  static void
  add_to_reclaim_list(details::hazard_data::node_pointer<_Ty> node) {
    auto *new_node = new details::hazard_data::reclaim_pointer<_Ty>(node);
    details::hazard_data::reclaim_pointer<_Ty> expected =
        hazard_data::reclaim_head<_Ty>.load();
    do {
      new_node->next = expected;
    } while (!hazard_data::reclaim_head<_Ty>.compare_exchange_weak(expected,
                                                                   new_node));
  }

  template <typename _Ty>
  static void try_reclaim(details::hazard_data::node_pointer<_Ty> new_node) {
    add_to_reclaim_list<_Ty>(new_node);
  }
  template <typename _Ty> static void auto_remove_from_reclaim_list() {
    // It will not influence the pervious one
    details::hazard_data::reclaim_pointer<_Ty> head =
        hazard_data::reclaim_head<_Ty>.exchange(nullptr);
    while (head) {
      details::hazard_data::reclaim_pointer<_Ty> next = head->next;
      if (!check_reference(head->node)) {
        delete head;
      } else {
        add_to_reclaim_list<_Ty>(head);
      }
      head = next;
    }
  }

protected:
  [[nodiscard]] hazard_pointer *find_unoccupancy() {
    for (std::size_t index = 0; index < MAX_HAZARD_NUMBER; ++index) {
      std::thread::id id{}; // default state
      if (hazard_data::hazard_pointers[index].id.compare_exchange_strong(
              id, std::this_thread::get_id())) {
        return &hazard_data::hazard_pointers[index];
      }
    }

    throw std::runtime_error("Insufficent Array Size!");
  }

private:
  hazard_pointer *hp{nullptr};
};

[[nodiscard]]
std::atomic<void *> &get_hazard_pointer_for_current_thread();
} // namespace details
} // namespace concurrency

template <typename _Ty> class concurrency::ConcurrentStack {

  friend class details::hazard_data;

  struct Node {
    Node() : data(nullptr), next(nullptr) {}
    Node(const _Ty &value)
        : data(std::make_shared<_Ty>(value)), next(nullptr) {}
    Node(_Ty &&value)
        : data(std::make_shared<_Ty>(std::move(value))), next(nullptr) {}
    std::shared_ptr<_Ty> data;
    Node *next;
  };

  ConcurrentStack(const ConcurrentStack &) = delete;
  ConcurrentStack &operator=(const ConcurrentStack &) = delete;

public:
  ConcurrentStack() : m_size(0), m_head(nullptr) {}

public:
  std::size_t size() const { return m_size.load(); }

  void push(const _Ty &value) {
    Node *new_node = new Node(value);
    __push(new_node);
  }

  void push(_Ty &&value) {
    Node *new_node = new Node(std::move(value));
    __push(new_node);
  }

  std::optional<std::shared_ptr<_Ty>> pop() { return __pop_by_smart_ptr(); }

protected:
  void __push(Node *new_node) {
    if (!new_node)
      return;
    Node *expected = m_head.load();
    do {
      new_node->next = expected;
    } while (!m_head.compare_exchange_weak(expected, new_node));

    m_size++;
  }

  std::optional<std::shared_ptr<_Ty>> __pop_by_smart_ptr() {
    Node *expected = m_head.load();
    while (true) {
      if (!expected)
        return std::nullopt;
      Node *next_node = expected->next;
      if (m_head.compare_exchange_weak(expected, next_node)) {
        --m_size;
        std::unique_ptr<Node> __ext_life_length(expected);
        return __ext_life_length->data;
      }
    }
    return std::nullopt;
  }

  std::optional<std::shared_ptr<_Ty>> __pop_by_hazard_ptr() {
    Node *expected = m_head.load();
    auto &hp = details::get_hazard_pointer_for_current_thread();
    do {
      do {
        expected = m_head.load();
        if (!expected)
          return std::nullopt;
        hp.store(expected);
      } while (m_head.load() != expected);

    } while (expected &&
             !m_head.compare_exchange_weak(expected, expected->next));

    auto res = expected->data;

    hp.store(nullptr); //

    /*no reference at all, delete it!*/
    if (!details::hazard_manager::check_reference(expected)) {
      delete expected;
    } else {
      details::hazard_manager::try_reclaim(expected);
    }

    /*clear reclaim list according to hazard pointers array*/
    details::hazard_manager::auto_remove_from_reclaim_list<_Ty>();
    return res;
  }

private:
  std::atomic<Node *> m_head{nullptr};
  std::atomic<std::size_t> m_size;
};

#endif //_STACK_LOCKFREE_HPP
