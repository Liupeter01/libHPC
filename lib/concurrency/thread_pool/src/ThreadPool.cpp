#include <ThreadPool.hpp>

ThreadPool::ThreadPool()
          :ThreadPool(std::thread::hardware_concurrency())
{
}

ThreadPool::ThreadPool(std::size_t threads) {
          m_threads = (threads <= 1 ? 2 : threads);
          start();
}

ThreadPool::~ThreadPool() {
          stop();
}

void ThreadPool::start() {
          m_stop.store(false);          //flag set

          for (std::size_t thread = 0; thread < m_threads; ++thread) {
                   m_pool.emplace_back(&ThreadPool::logicRunner, this);
          }
}

void ThreadPool::logicRunner(){
          while (!m_stop) {
                    TaskType task;
                    std::unique_lock<std::mutex> _lckg(m_mtx);   //create unique_lock
                    m_cv.wait(_lckg, [this]() {return m_stop || !m_queue.empty(); });
                    if (m_queue.empty())
                              return;

                    if (m_stop) {
                              while (!m_queue.empty()) {
                                        task = std::move(m_queue.front());
                                        m_queue.pop();

                                        /*start to execute*/
                                        task();
                              }

                              return;
                    }
                    
                    task = std::move(m_queue.front());
                    m_queue.pop();
                    _lckg.unlock();
                    
                    /*start to execute*/
                    task();
          }
}

void ThreadPool::stop() {
          m_stop.store(true);           //flag set
          m_cv.notify_all();

          /*join all threads*/
          for (auto& thread : m_pool) {
                    if (thread.joinable()) {
                              thread.join();
                    }
          }
}

