#include <stack_lockfree.hpp>

concurrency::details::hazard_pointer
concurrency::details::hazard_data::hazard_pointers[MAX_HAZARD_NUMBER]{};

template<typename _Ty>
std::atomic<concurrency::details::hazard_data::ReclaimNode<_Ty>*>
concurrency::details::hazard_data::reclaim_head{ nullptr };

[[nodiscard]]
std::atomic<void*>& 
concurrency::details::get_hazard_pointer_for_current_thread() {
          thread_local static hazard_manager manager;
          return manager.get_pointer();
}
