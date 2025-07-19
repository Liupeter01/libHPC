#pragma once
#ifndef _STACK_LK_HPP
#define _STACK_LK_HPP
#include <condition_variable>
#include <memory>
#include <mutex>
#include <stack>

namespace concurrency {
template <typename _Ty> class StackLk;
}

template <typename _Ty> class concurrency::StackLk {
public:
  StackLk &operator=(const StackLk &) = delete;

public:
  StackLk(const StackLk &o) {
    std::lock_guard<std::mutex> _lckg(m_mtx);
    m_data = o.m_data;
  }
  void push(const _Ty &value) {
    {
      std::lock_guard<std::mutex> _lckg(m_mtx);
      m_data.push(value);
    }
    m_cv.notify_one();
  }
  void push(_Ty &&value) {
    {
      std::lock_guard<std::mutex> _lckg(m_mtx);
      m_data.push(std::move(value));
    }
    m_cv.notify_one();
  }
  std::shared_ptr<_Ty> wait_and_pop() {
    std::unique_lock<std::mutex> _lckg(m_mtx);
    m_cv.wait(_lckg, [this]() { return !m_data.empty(); });
    auto res = std::make_shared<_Ty>(std::move(m_data.top()));
    m_data.pop();
    return res;
  }
  void wait_and_pop(_Ty &value) {
    std::unique_lock<std::mutex> _lckg(m_mtx);
    m_cv.wait(_lckg, [this]() { return !m_data.empty(); });
    value = std::move(m_data.top());
    m_data.pop();
  }
  bool try_pop(_Ty &value) {
    std::lock_guard<std::mutex> _lckg(m_mtx);
    if (m_data.empty()) {
      return false;
    }
    value = std::move(m_data.top());
    m_data.pop();
    return true;
  }
  bool empty() const {
    std::lock_guard<std::mutex> _lckg(m_mtx);
    return m_data.empty();
  }

private:
  std::stack<_Ty> m_data;
  mutable std::mutex m_mtx;
  std::condition_variable m_cv;
};

#endif //_STACK_LK_HPP
