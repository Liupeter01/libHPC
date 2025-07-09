#pragma once
#ifndef _QUEUE_LK_HPP
#define _QUEUE_LK_HPP
#include <mutex>
#include <queue>
#include <condition_variable>
#include <memory>

namespace concurrency {
          template<typename _Ty>
          class QueueLk;

          template<typename _Ty>
          class ConcurrentQueueLk;
}

template<typename _Ty>
class concurrency::QueueLk {
public:
          QueueLk(const QueueLk& o) {
                    std::lock_guard<std::mutex> _lckg(m_mtx);
                    m_data = o.m_data;
          }

          void push(const _Ty& value) {
                    auto ptr = std::make_shared<_Ty>(value);
                    {
                              std::lock_guard<std::mutex> _lckg(m_mtx);
                              m_data.push(ptr);
                    }
                    m_cv.notify_one();
          }

          void push(_Ty&& value) {
                    auto value = std::make_shared<_Ty>(std::move(value));
                    {
                              std::lock_guard<std::mutex> _lckg(m_mtx);
                              m_data.push(ptr);
                    }
                    m_cv.notify_one();
          }

          std::shared_ptr<_Ty> wait_and_pop() {
                    std::unique_lock<std::mutex> _lckg(m_mtx);
                    m_cv.wait(_lckg, [this]() { return !m_data.empty(); });
                    auto res = m_data.front();
                    m_data.pop();
                    return res;
          }

          void wait_and_pop(_Ty& value) {
                    std::unique_lock<std::mutex> _lckg(m_mtx);
                    m_cv.wait(_lckg, [this]() { return !m_data.empty(); });
                    value = std::move(*m_data.front());
                    m_data.pop();
          }

          bool try_pop(_Ty& value) {
                    std::lock_guard<std::mutex> _lckg(m_mtx);
                    if (m_data.empty()) {
                              return false;
                    }
                    value = std::move(*m_data.front());
                    m_data.pop();
                    return true;
          }

          bool empty() const {
                    std::lock_guard<std::mutex> _lckg(m_mtx);
                    return m_data.empty();
          }

private:
          std::queue<std::shared_ptr<_Ty>> m_data;
          mutable std::mutex m_mtx;
          std::condition_variable m_cv;
};

template<typename _Ty>
struct concurrency::ConcurrentQueueLk {
private:
          struct Node {
                    Node() : value(nullptr), next(nullptr)
                    {
                    }
                    Node(const _Ty& _value)
                              : value(std::make_shared<_Ty>(_value)), next(nullptr)
                    {
                    }

                    std::shared_ptr<_Ty> value;
                    std::unique_ptr<Node> next;
          };

          Node* get_tail() {
                    std::lock_guard<std::mutex> _lckg_push(m_tail_mtx);
                    return m_tail;
          }

public:
          ConcurrentQueueLk()
                    :m_head(std::make_unique< Node>()), m_tail(m_head.get())
          {
          }

          void push(const _Ty& value){
                    auto data = std::make_shared<_Ty>(value);
                    auto new_node = std::make_unique<Node>();
                    auto next_ptr = new_node.get();
                    {
                              std::lock_guard<std::mutex> _lckg_push(m_tail_mtx);
                              m_tail->value = data;
                              m_tail->next = std::move(new_node);
                              m_tail = next_ptr;
                    }
                    m_cv.notify_one();
           }


          void pop(_Ty& value) {
                    std::unique_lock<std::mutex>_lckg_pop(m_head_mtx);
                    m_cv.wait(_lckg_pop, [this]() {return get_tail() != m_head.get();  });
                    value = std::move(*m_head->value);
                    m_head = std::move(m_head->next);
          }
          std::shared_ptr<_Ty> pop() {
                    std::unique_lock<std::mutex>_lckg_pop(m_head_mtx);
                    m_cv.wait(_lckg_pop, [this]() {return get_tail() != m_head.get();  });
                    auto res = m_head->value;
                    m_head = std::move(m_head->next);
                    return res;
          }


private:
          struct Node {
                    Node()  : value(nullptr), next(nullptr)
                    { }
                    Node(const _Ty& _value) 
                              : value(std::make_shared<_Ty>(_value)), next(nullptr) 
                    {}

                    std::shared_ptr<_Ty> value;
                    std::unique_ptr<Node> next;
          };

          std::mutex m_head_mtx;
          std::mutex m_tail_mtx;

          std::unique_ptr<Node> m_head;
          Node* m_tail;

          std::condition_variable m_cv;
};


#endif // _QUEUE_LK_HPP