#pragma once
#ifndef _QUEUE_LOCKFREE_HPP
#define _QUEUE_LOCKFREE_HPP
#include <atomic>
#include <memory>
#include <optional>

namespace concurrency {
template <typename _Ty> class ConcurrentQueue;
}

template <typename _Ty> class concurrency::ConcurrentQueue {

          struct ReferenceNode;
          struct ref_counter_packed {
                    ref_counter_packed() : thread_ref_counter(0), head_and_tail_ref_counter(2) {}
                    std::intptr_t thread_ref_counter: 30;			//how many threads are referencing this node?
                    unsigned int head_and_tail_ref_counter : 2;	//m_head or m_tail or next(pointer) are referencing on this!

                    const bool both_zero() const{
                              return !(thread_ref_counter && head_and_tail_ref_counter);
                    }
          };

          struct Node {
                    Node() : data(nullptr), inner_ref_counter() {
                              next.node = nullptr;
                              next.thread_ref_counter = 0;
                    }
                    Node(const _Ty& value) : data(std::make_unique<_Ty>(value)), inner_ref_counter() {
                              next.node = nullptr;
                              next.thread_ref_counter = 0;
                    }
                    Node(_Ty&& value) : data(std::make_unique<_Ty>(std::move(value))), inner_ref_counter() {
                              next.node = nullptr;
                              next.thread_ref_counter = 0;
                    }

                    ReferenceNode next;
                    std::atomic<_Ty*> data;
                    std::atomic<ref_counter_packed> inner_ref_counter;

                    void sync_threads_ref(const std::intptr_t any_other_threads) {
                              ref_counter_packed old_inner_ref =inner_ref_counter.load();
                              ref_counter_packed updated_inner_ref{};
                              do {
                                        updated_inner_ref = old_inner_ref;
                                        --updated_inner_ref.head_and_tail_ref_counter;
                                        updated_inner_ref.thread_ref_counter += any_other_threads;
                              } while (!inner_ref_counter.compare_exchange_weak(old_inner_ref, updated_inner_ref));
                    }

                    void release_curr_thread_ref() {
                              ref_counter_packed old_inner_ref = inner_ref_counter.load();
                              ref_counter_packed updated_inner_ref{};
                              do {
                                        updated_inner_ref = old_inner_ref;
                                        --updated_inner_ref.thread_ref_counter;
                              } while (!inner_ref_counter.compare_exchange_weak(old_inner_ref, updated_inner_ref));
                    }

                    bool remove_data() {
                              if (inner_ref_counter.load().both_zero()) {
                                        delete data.load();
                                        return true;
                              }
                              return false;
                    }
          };

          struct ReferenceNode {
                    ReferenceNode() : thread_ref_counter(1), node(nullptr) {}
                    ReferenceNode(const _Ty& value) : thread_ref_counter(1), node(new Node(value)) {}
                    ReferenceNode(_Ty&& value) : thread_ref_counter(1), node(new Node(std::move(value))) {}
                    std::intptr_t thread_ref_counter;	//how many threads are referencing this node?
                    Node* node;
          };

          ConcurrentQueue(const ConcurrentQueue&) = delete;
          ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;

public:
          ConcurrentQueue() {
                    ReferenceNode init_node;
                    init_node.node = new Node;
                    init_node.thread_ref_counter = 1;

                    m_head.store(init_node, std::memory_order_release);
                    m_tail.store(init_node, std::memory_order_release);
          }
          virtual ~ConcurrentQueue() {
                    clear();
          }

public:
          void clear() {
                    while (!empty()) {
                              auto res = pop();
                              (void)res;
                    }
          }

          const bool empty() const { 
                    ReferenceNode old_head = m_head.load();
                    return ((old_head.node == m_tail.load().node) ? true : false);
          }

          /*Maybe Approximate!*/
          const std::size_t size() const { return m_size; }

          void push(_Ty&& value) {
                    __push(std::make_unique<_Ty>(std::move(value)));
          }
          void push(const _Ty& value) {
                    __push(std::make_unique<_Ty>(value));
          }

          [[nodiscard]]
          std::optional<std::unique_ptr<_Ty>> 
          pop() {
                    return __pop();
          }

protected:
          void __push(std::unique_ptr<_Ty> value) {
                    ReferenceNode next_node;
                    next_node.node = new Node;
                    next_node.thread_ref_counter = 1;

                    ReferenceNode old_tail = m_tail.load();
                    for (;;) {
                              old_tail  = __increase_ref_rmw(m_tail, old_tail);

                              [[maybe_unused]] _Ty* old_data{ nullptr };
                              if (old_tail.node->data.compare_exchange_strong(old_data, value.get())) {

                                        old_tail.node->next = next_node;

                                        //get newest m_tail reference counter!
                                        old_tail = m_tail.exchange(next_node);

                                        /*Boardcast Current Global Threads Reference Via Atomic*/
                                        __sync_threads_ref(old_tail);

                                        value.release();    //release resource

                                        ++m_size;
                                        break;
                              }

                              //other failed thread: release reference count!
                              __release_curr_thread_ref(old_tail);
                    }
          }

          [[nodiscard]]
          std::optional<std::unique_ptr<_Ty>>  __pop() {
                    ReferenceNode old_head = m_head.load();
                    for (;;) {
                              old_head = __increase_ref_rmw(m_head, old_head);
                              if (!old_head.node)  return std::nullopt;

                              //empty?
                              if (old_head.node == m_tail.load().node) {
                                        __release_curr_thread_ref(old_head);
                                        return std::nullopt;
                              }

                              if (m_head.compare_exchange_strong(old_head, old_head.node->next)) {

                                        std::unique_ptr<_Ty> res = std::unique_ptr<_Ty>(old_head.node->data.exchange(nullptr));

                                        __remove_node_from_heap(old_head);

                                        --m_size;
                                        return res;
                              }

                              __release_curr_thread_ref(old_head);
                    }
          }

private:
          [[nodiscard]]
          static
                    ReferenceNode
                    __increase_ref_rmw(std::atomic<ReferenceNode>& main_node, 
                              ReferenceNode& old) {
                    ReferenceNode new_ref;
                    do {
                              new_ref = old;
                              new_ref.thread_ref_counter += 1;
                    } while (!main_node.compare_exchange_weak(old, new_ref));
                    return new_ref;
          }

          static void __sync_threads_ref(ReferenceNode& old) {
                    if (!old.node) return;
                    const std::intptr_t any_other_threads = old.thread_ref_counter - 2;        //-2!
                    old.node->sync_threads_ref(any_other_threads);
          }

          static void __release_curr_thread_ref(ReferenceNode& old) {
                    if (!old.node) return;
                    old.node->release_curr_thread_ref();
          }
          static const bool __remove_data(ReferenceNode& old) {
                    if (!old.node) return false;
                    return old.node->remove_data();
          }
          static const bool __remove_node_from_heap(ReferenceNode& old) {
                    __sync_threads_ref(old);
                    return __remove_data(old);
          }

private:
          std::atomic<std::size_t> m_size;
          std::atomic<ReferenceNode> m_head;
          std::atomic<ReferenceNode> m_tail;
};

#endif // _QUEUE_LOCKFREE_HPP
