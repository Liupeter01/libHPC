#pragma once
#ifndef _THREADPOOL_HPP_
#define _THREADPOOL_HPP_
#include <RetType.hpp>
#include <Singleton.hpp>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool : public Singleton<ThreadPool> {
  friend class Singleton<ThreadPool>;
  using TaskType = std::packaged_task<void()>;
  ThreadPool();
  ThreadPool(std::size_t threads);

public:
  virtual ~ThreadPool();
  void stop();
  template <typename Func, typename... Args>
  std::future<RetValue<Func, Args...>> commit(Func &&func, Args &&...args) {
    using RetType = RetValue<Func, Args...>;
    using FinalType = std::packaged_task<RetType()>;

    if (m_stop) {
      return {};
    }

    auto task = std::make_shared<FinalType>(
        std::bind(std::forward<Func>(func), std::forward<Args>(args)...));

    auto ret = task->get_future();
    {
      std::lock_guard<std::mutex> _lckg(m_mtx);
      m_queue.emplace([task]() { (*task)(); });
    }
    m_cv.notify_one();
    return ret;
  }

protected:
  void start();
  void logicRunner();

private:
  std::size_t m_threads;
  std::atomic_bool m_stop;
  std::mutex m_mtx;
  std::condition_variable m_cv;
  std::queue<TaskType> m_queue;
  std::vector<std::thread> m_pool;
};

#endif //_THREADPOOL_HPP_
