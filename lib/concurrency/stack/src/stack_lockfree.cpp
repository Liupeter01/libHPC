#include <stack_lockfree.hpp>

namespace concurrency::details {

          inline hazard_pointer hazard_data::hazard_pointers[MAX_HAZARD_NUMBER]{};

          [[nodiscard]]
          std::atomic<void*>& get_hazard_pointer_for_current_thread() {
                    thread_local static hazard_manager manager;
                    return manager.get_pointer();
          }
}
