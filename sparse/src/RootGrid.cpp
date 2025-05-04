#include <RootGrid.hpp>

template <typename _Ty, typename _Layout>
std::optional<_Ty> 
sparse::RootGrid<_Ty, _Layout>::read(std::intptr_t x, std::intptr_t y) const {
          return _read(m_root, x, y);
}

template <typename _Ty, typename _Layout>
template<typename Node>
std::optional<_Ty> 
sparse::RootGrid<_Ty, _Layout>::_read(const Node& node, std::intptr_t x, std::intptr_t y) {
          //if it is denseblock, then it should be the last node point!
          if constexpr (Node::is_leaf) {
                    return node.read(x, y);
          }
          else {
                    const auto [new_x, new_y] = generateCoord<Node>(x, y);
                    /*if it is hash/pointer block, then find the next block and do recursion!*/
                    auto next_node = node.read(new_x, new_y);

                    if (!next_node.has_value()) return std::nullopt;
                    //you have to pass global coordinates to the function!
                    return _read(next_node.value(), x, y);
          }
}

template <typename _Ty, typename _Layout>
template<typename Node>
void 
sparse::RootGrid<_Ty, _Layout>::_write(Node& node, std::intptr_t x, std::intptr_t y, const _Ty& value) {
          //if it is denseblock, then it should be the last node point!
          if constexpr (Node::is_leaf) {
                    node.write(x, y, value);
          }
          else {
                    const auto [new_x, new_y] = generateCoord<Node>(x, y);
                    /*if it is hash/pointer block, then find the next block and do recursion!*/
                    auto& next_node = node.touch_pointer(new_x, new_y);
                    _write(next_node.get(), x, y, value); 
          }
}

template <typename _Ty, typename _Layout>
void 
sparse::RootGrid<_Ty, _Layout>::write(std::intptr_t x, 
                                                                 std::intptr_t y, 
                                                                  const _Ty& value) {

          _write(m_root, x, y, value);
}

template <typename _Ty, typename _Layout>
template<typename Func>
void 
sparse::RootGrid<_Ty, _Layout>::foreach(Func&& func) {

}