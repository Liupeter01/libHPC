#include <cuda_hierarchical_exclusive_scan_localT_1024.cuh>

namespace sort {
          namespace gpu {
                    namespace radix {
                              namespace details {

                                        // ================================================================
                                        // build_level_layout_and_allocate
                                        //
                                        // Build a hierarchical layout for tile-sum levels and allocate
                                        // one packed device buffer to store all levels.
                                        //
                                        // Each hierarchy level represents a per-bin array of tile SUMS:
                                        //
                                        //   - level0: tile sums of localT
                                        //   - level1: tile sums of level0
                                        //   - level2: tile sums of level1
                                        //   - ...
                                        //
                                        // Layout invariants:
                                        //   - level[k].real = ceil(level[k-1].real / 1024)
                                        //   - The hierarchy shrinks by a factor of 1024 per level
                                        //   - The top level is guaranteed to satisfy: real <= 1024
                                        //
                                        // This guarantees that:
                                        //   - The top level can always be scanned by a single CTA
                                        //   - All lower levels can be converted into OFFSETS via downsweep
                                        //
                                        // Memory layout:
                                        //   - All levels are packed into one contiguous buffer:
                                        //       big_buffer = [ level0 | level1 | level2 | ... ]
                                        //   - Each level is laid out as:
                                        //       [BinSize][stride], where stride >= real and is aligned
                                        //
                                        // Inputs:
                                        //   BinSize     : number of bins (e.g., 256)
                                        //   local_real  : number of valid elements per bin in localT
                                        //   align_stride: stride alignment in elements (e.g., 32)
                                        //
                                        // Output:
                                        //   levels[] describing each hierarchy level:
                                        //     levels[0]      -> level0 (tile sums of localT)
                                        //     levels[1]      -> tile sums of level0
                                        //     ...
                                        //     levels.back()  -> top level (real <= 1024)
                                        //
                                        // IMPORTANT:
                                        //   - This function only builds layout and allocates memory.
                                        //   - It does NOT perform any scan or modification of data.
                                        // ================================================================

                                        std::vector<LevelDesc> build_level_layout_and_allocate(
                                                  thrust::device_vector<uint32_t>& big_buffer,
                                                  uint32_t BinSize,
                                                  uint32_t local_real,
                                                  uint32_t align_stride ) {  // alignment in elements
                                                  std::vector<LevelDesc> levels;

                                                  // Number of tiles in level0 (one tile = 1024 elements of localT)
                                                  const uint32_t level0_real = div_up_u32(local_real, 1024u);

                                                  // Compute logical lengths for each hierarchy level:
                                                  //   r0 = level0_real
                                                  //   r1 = ceil(r0 / 1024)
                                                  //   ...
                                                  // Stop once r <= 1024 so top level fits in one CTA.
                                                  uint32_t real = level0_real;
                                                  std::vector<uint32_t> reals;
                                                  reals.reserve(8); // log_1024(local_real) <= 8 for any sane input

                                                  while (real > 1024) {
                                                            reals.push_back(real);
                                                            real = div_up_u32(real, 1024u);
                                                  }

                                                  // Top level: must fit into a single CTA (<= 1024 elements)
                                                  reals.push_back(real);

                                                  // Debug invariant (highly recommended)
#ifndef NDEBUG
                                                  assert(reals.back() <= 1024);
#endif

                                                  // Compute per-level strides (aligned) and prefix bases
                                                  std::vector<uint32_t> strides(reals.size());
                                                  std::vector<uint32_t> bases(reals.size());

                                                  uint32_t sum_stride = 0;
                                                  for (size_t i = 0; i < reals.size(); ++i) {
                                                            strides[i] = align_up_u32(reals[i], align_stride);
                                                            bases[i] = sum_stride * BinSize;
                                                            sum_stride += strides[i];
                                                  }

                                                  // Allocate one packed buffer for all levels:
                                                  // total elements = BinSize * sum(level_stride)
                                                  big_buffer.resize(BinSize * sum_stride);
                                                  uint32_t* base_ptr = big_buffer.data().get();

                                                  // Fill level descriptors
                                                  for (size_t i = 0; i < reals.size(); ++i) {
                                                            LevelDesc d;
                                                            d.ptr = base_ptr;
                                                            d.base = bases[i];
                                                            d.stride = strides[i];
                                                            d.real = reals[i];
                                                            levels.push_back(d);
                                                  }

                                                  return levels;
                                        }

                                        // ================================================================
                                        // hierarchical_exclusive_scan_localT_1024
                                        //
                                        // Perform a full EXCLUSIVE scan over localT using a hierarchical
                                        // 1024-element tile-based scan.
                                        //
                                        // Conceptual model (per bin):
                                        //
                                        //   localT (elements)
                                        //      |
                                        //      |  [tile-local exclusive scan]
                                        //      v
                                        //   localT (tile-local offsets)           <-- still missing cross-tile offsets
                                        //      |
                                        //      |  emit tile SUMS
                                        //      v
                                        //   level0 (tile SUMS)
                                        //      |
                                        //      |  hierarchical upsweep
                                        //      v
                                        //   levelN (top-level SUMS, real <= 1024)
                                        //      |
                                        //      |  exclusive scan (single CTA)
                                        //      v
                                        //   levelN (top-level OFFSETS)
                                        //      |
                                        //      |  downsweep (pure offset propagation)
                                        //      v
                                        //   level0 (tile OFFSETS)
                                        //      |
                                        //      |  final apply
                                        //      v
                                        //   localT (full EXCLUSIVE scan)
                                        //
                                        // ------------------------------------------------------------
                                        //
                                        // IMPORTANT DESIGN INVARIANTS:
                                        //
                                        // 1) Only ONE level is explicitly scanned as "SUM -> OFFSET":
                                        //      - The TOP level (levels.back())
                                        //      - Guaranteed to have real <= 1024 by construction
                                        //
                                        // 2) All lower levels (including level0) are NEVER directly scanned.
                                        //      - They are converted from SUMS to OFFSETS purely by downsweep.
                                        //
                                        // 3) level0.real may be >1024 (e.g., 1107), and this is INTENTIONAL.
                                        //      - As long as there exists a higher level,
                                        //        level0 offsets are obtained correctly via propagation.
                                        //
                                        // 4) No thrust / global scan is used at any level.
                                        //      - All transitions are explicit and deterministic.
                                        //
                                        // ------------------------------------------------------------
                                        //
                                        // Parameters:
                                        //   localT        : [BinSize][local_stride] element-level data
                                        //   level_buffer  : packed buffer holding all hierarchy levels
                                        //   levels        : layout descriptors from build_level_layout_and_allocate
                                        //   BinSize       : number of bins
                                        //   local_stride  : per-bin stride of localT
                                        //   local_real    : number of valid elements per bin
                                        //   stream        : CUDA stream
                                        // ================================================================

                                        void hierarchical_exclusive_scan_localT_1024(
                                                  uint32_t* localT,
                                                  thrust::device_vector<uint32_t>& level_buffer,
                                                  const std::vector<LevelDesc>& levels,
                                                  uint32_t BinSize,
                                                  uint32_t local_stride, // per-bin stride of localT
                                                  uint32_t local_real,   // number of valid elements per bin
                                                  cudaStream_t stream) {

                                                  // Number of 1024-element tiles in localT
                                                  const uint32_t level0_real = div_up_u32(local_real, 1024u);

                                                  // Base pointer of packed level buffer
                                                  uint32_t* Lbase = level_buffer.data().get();

                                                  dim3 block(32, 32, 1); // 1024 threads per CTA (32 warps)

                                                  // -----------------------------
                                                  // Upsweep phase
                                                  //
                                                  // For each level:
                                                  //   - Perform a tile-local EXCLUSIVE scan IN-PLACE
                                                  //   - Emit one SUM per tile into the next level
                                                  //
                                                  // After this phase:
                                                  //   - localT contains tile-local EXCLUSIVE scans
                                                  //   - levels[k] contain tile SUMS (NOT offsets yet)
                                                  // -----------------------------
                                                  {
                                                            // Round 0: localT -> level0 sums
                                                            dim3 grid(level0_real, BinSize, 1);
                                                            uint32_t* level0_ptr = Lbase + levels[0].base;

                                                            v1::kernel_scan_tile1024_exclusive_inplace_and_emit_tile_sum<32>
                                                                      << <grid, block, 0, stream >> > (
                                                                                localT,
                                                                                level0_ptr,
                                                                                local_stride,
                                                                                local_real,
                                                                                levels[0].stride,
                                                                                level0_real
                                                                                );

                                                            // Higher rounds: level(k-1) -> level(k)
                                                            for (std::size_t k = 1; k < levels.size(); ++k) {
                                                                      const auto& in = levels[k - 1];
                                                                      const auto& out = levels[k];

                                                                      dim3 grid(div_up_u32(in.real, 1024u), BinSize, 1);

                                                                      v1::kernel_scan_tile1024_exclusive_inplace_and_emit_tile_sum<32>
                                                                                << <grid, block, 0, stream >> > (
                                                                                          Lbase + in.base,
                                                                                          Lbase + out.base,
                                                                                          in.stride,
                                                                                          in.real,
                                                                                          out.stride,
                                                                                          out.real
                                                                                          );
                                                            }
                                                  }


                                                  // -----------------------------
                                                  // Convert TOP-level SUMS -> OFFSETS
                                                  //
                                                  // The top level is guaranteed to satisfy:
                                                  //   top.real <= 1024
                                                  //
                                                  // Therefore, a single CTA per bin is sufficient to perform
                                                  // a correct EXCLUSIVE scan.
                                                  //
                                                  // This is the ONLY place where a SUM -> OFFSET scan is performed.
                                                  // -----------------------------
                                                  {
                                                            const auto& top = levels.back();

                                                            // By construction (layout), top.real <= 1024
                                                            dim3 grid(1, BinSize, 1);
                                                            dim3 blk(32, 32, 1);

                                                            v1::kernel_exclusive_scan_upto1024_per_bin<32>
                                                                      << <grid, blk, 0, stream >> > (
                                                                                Lbase + top.base,
                                                                                top.stride,
                                                                                top.real
                                                                                );
                                                  }

                                                  // -----------------------------
                                                  // Downsweep phase
                                                  //
                                                  // Propagate OFFSETS from higher levels to lower levels.
                                                  //
                                                  // For each level pair (hi -> lo):
                                                  //   lo[i] += hi[i / 1024]
                                                  //
                                                  // Notes:
                                                  //   - hi already contains EXCLUSIVE OFFSETS
                                                  //   - lo currently contains tile-local EXCLUSIVE values
                                                  //   - This step does NOT perform any scan
                                                  //   - It only applies offsets
                                                  //
                                                  // After downsweep completes:
                                                  //   - level0 contains correct tile OFFSETS
                                                  // -----------------------------

                                                  if (levels.size() >= 2) {
                                                            for (std::size_t k = levels.size(); k-- > 1;) {
                                                                      const auto& hi = levels[k];
                                                                      const auto& lo = levels[k - 1];

                                                                      dim3 grid(div_up_u32(lo.real, 1024u), BinSize, 1);

                                                                      v1::kernel_apply_tile_offsets_to_element_offsets_1024<32>
                                                                                << <grid, block, 0, stream >> > (
                                                                                          Lbase + lo.base,
                                                                                          Lbase + hi.base,
                                                                                          lo.stride, lo.real,
                                                                                          hi.stride, hi.real
                                                                                          );
                                                            }
                                                  }

                                                  // ------------------------------------------------------------
                                                  // Final apply
                                                  //
                                                  // Combine:
                                                  //   - tile-local EXCLUSIVE results in localT
                                                  //   - tile-level OFFSETS stored in level0
                                                  //
                                                  // Formula:
                                                  //   localT[bin][i] += level0[bin][ i >> 10 ]
                                                  //
                                                  // After this step:
                                                  //   localT contains a full EXCLUSIVE scan over all elements.
                                                  // ------------------------------------------------------------

                                                  {
                                                            dim3 grid(div_up_u32(local_real, 1024u), BinSize, 1);

                                                            v1::kernel_apply_level0_to_localT_1024<32>
                                                                      << <grid, block, 0, stream >> > (
                                                                                localT,
                                                                                Lbase + levels[0].base,
                                                                                local_stride,
                                                                                local_real,
                                                                                levels[0].stride,
                                                                                level0_real
                                                                                );
                                                  }
                                        }

                              } // namespace details
                    } // namespace radix
          } // namespace gpu
} // namespace sort