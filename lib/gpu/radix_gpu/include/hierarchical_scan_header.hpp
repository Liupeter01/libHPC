#pragma once
#ifndef _HIERARCHICAL_SCAN_HEADER_HPP_
#define _HIERARCHICAL_SCAN_HEADER_HPP_
#include <common.hpp>

namespace sort {
          namespace gpu {
                    namespace radix {
                              namespace details {
                                        // ================================================================
                                        // LevelDesc / packed buffer layout
                                        //
                                        // We store hierarchy levels in one big packed device buffer:
                                        //   level_buffer shape: [BinSize][ sum(level_strides) ]
                                        //
                                        // Each level k is a view:
                                        //   level[k][bin][0 .. real-1], with stride >= real (aligned)
                                        //
                                        // The `real` is the logical length of that level per bin.
                                        // The `stride` is padded/aligned length per bin for coalescing/alignment.
                                        //
                                        // Example (local_real -> level0_real -> level1_real -> ...):
                                        //   level0_real = ceil(local_real / 1024)
                                        //   level1_real = ceil(level0_real / 1024)
                                        //   ...
                                        //
                                        // In the "correct version" you currently use:
                                        //   stop when real <= 1024
                                        // meaning: the topmost level fits into a single CTA scan kernel if desired.
                                        // ================================================================
                                        struct LevelDesc {
                                                  uint32_t* ptr;     // base pointer (beginning of whole buffer)
                                                  uint32_t base;     // offset (elements) from ptr to this level's bin0 start
                                                  uint32_t stride;   // per-bin stride in elements (>= real)
                                                  uint32_t real;     // valid length per bin
                                        };
                              }
                    }
          }
}

#endif //_HIERARCHICAL_SCAN_HEADER_HPP_