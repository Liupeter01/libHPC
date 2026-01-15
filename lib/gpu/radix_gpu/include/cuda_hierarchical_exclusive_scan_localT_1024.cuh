#pragma once
#ifndef _CUDA_HIERARCHICAL_EXCLUSIVE_SCAN_CUH_
#define _CUDA_HIERARCHICAL_EXCLUSIVE_SCAN_CUH_
#include <common.hpp>
#include <hierarchical_scan_header.hpp>
#include <thrust/device_vector.h>
#include <vector>

// ================================================================
// Hierarchical Exclusive Scan over localT (per bin), tile size = 1024
//
// Problem setting
// ---------------
// We have a 2D layout:
//
//   localT[bin][i],   bin in [0, BinSize), i in [0, local_real)
//
// For radix sort, localT is typically the per-block histogram (transposed):
//   localT[bin][block] = count of keys in block that fall into `bin`
//
// Goal
// ----
// Transform each bin-row into a full EXCLUSIVE prefix sum:
//
//   localT[bin][i] = sum_{j < i} localT_original[bin][j]
//
// Constraints / motivation
// ------------------------
// local_real can be very large (millions of blocks).
// We want a scalable GPU scan without relying on one giant temporary allocation
// per invocation, and we want deterministic behavior for padded regions.
//
// Strategy: hierarchy of 1024-element tiles
// -----------------------------------------
// We scan in a hierarchy where each CTA processes exactly one tile (1024 elems):
//
//   tile size = 1024 = 32 warps * 32 lanes
//
// Upsweep:
//   - round 0: scan each tile of localT in-place (tile-local exclusive scan),
//              and emit tile SUMs into level0
//   - round k: scan each tile of level(k-1) in-place, and emit tile SUMs into level(k)
//
// IMPORTANT semantic invariant after UPSWEEP:
//   - localT contains tile-local exclusive scan results (NOT global yet)
//   - each level[k] contains TILE SUMS (not offsets yet)
//   - Only levels that were used as "in" in some round got tile-local exclusive scan;
//     but their emitted outputs are still sums.
//
// Top conversion:
//   - At some top level, we must convert its tile SUMS into tile OFFSETS
//     (exclusive scan across tiles).
//   - Once top level becomes OFFSETS, we can propagate offsets downward.
//
// Downsweep:
//   - NO scan happens here.
//   - Only "apply" offsets downward:
//       level[k-1][i] += level[k][ i >> 10 ]
//     This converts level[k-1] from "tile sums" into "tile offsets" at that level.
//
// Final apply:
//   - localT already has tile-local exclusive results from round 0.
//   - Add level0 offsets:
//       localT[bin][i] += level0[bin][ i >> 10 ]
//
// Key design principle:
// ---------------------
// Down-sweep kernels NEVER perform scan.
// They are pure "offset propagation" operations.
//
// This is why correctness depends on a single fact:
//   - the highest level must be exclusive-scanned into OFFSETS (or handle level0-only case).
// ================================================================

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
                                                  uint32_t align_stride = 32// alignment in elements
                                        );

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
                                                  uint32_t local_real,   // numBlocks
                                                  cudaStream_t stream = 0
                                        );

                                        namespace v1 {

                                                  // ============================================================================
                                                     // Kernel 1: kernel_scan_tile1024_exclusive_inplace_and_emit_tile_sum
                                                     //
                                                     // Role in hierarchy:
                                                     //   * Bottom-up (upsweep) phase
                                                     //   * Converts ELEMENT-LEVEL data into:
                                                     //       - tile-local EXCLUSIVE scans (in-place)
                                                     //       - tile SUMS (one per 1024-element tile)
                                                     //
                                                     // What this kernel DOES:
                                                     //   For each (bin, tile):
                                                     //     1) Perform an EXCLUSIVE scan over exactly 1024 elements
                                                     //        (padding with 0 beyond in_real).
                                                     //     2) Write the tile-local exclusive results back to `in`.
                                                     //     3) Emit the TOTAL SUM of the tile into `tile_sum_out[bin][tile]`.
                                                     //
                                                     // What this kernel DOES NOT DO:
                                                     //   * It does NOT compute global offsets.
                                                     //   * It does NOT depend on higher-level offsets.
                                                     //   * It does NOT read or write outside its tile.
                                                     //
                                                     // Tile definition:
                                                     //   * One tile = 1024 elements
                                                     //   * Execution: 32 warps ¡Á 32 lanes (one CTA)
                                                     //
                                                     // Input / Output semantics:
                                                     //   in:
                                                     //     - Input : raw values (round 0) OR lower-level tile sums (round k>=1)
                                                     //     - Output: tile-local EXCLUSIVE scan results (in-place)
                                                     //
                                                     //   tile_sum_out:
                                                     //     - Receives the TOTAL SUM of each tile
                                                     //     - Logical length per bin: out_real = ceil(in_real / 1024)
                                                     //
                                                     // Padding rule:
                                                     //   * Elements with linear index >= in_real are treated as 0
                                                     //   * Padding does NOT affect correctness of sums or scans
                                                     //
                                                     // Memory layout:
                                                     //   * in           : [BinSize][in_stride]
                                                     //   * tile_sum_out : [BinSize][out_stride]
                                                     //   * Only indices < in_real / out_real are logically valid
                                                     //
                                                     // Launch configuration:
                                                     //   block = dim3(32, 32, 1)    // 1024 threads (one CTA per tile)
                                                     //   grid  = dim3(out_real, BinSize, 1)
                                                     // 
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
                                                  // 
                                                     // Correctness invariants:
                                                     //   * out_real == ceil(in_real / 1024)
                                                     //   * blockIdx.x < out_real
                                                     //   * Each CTA processes exactly ONE tile for ONE bin
                                                     // ============================================================================
                                                  template<std::size_t BlockSize = 32>
                                                  __global__ void kernel_scan_tile1024_exclusive_inplace_and_emit_tile_sum(
                                                            uint32_t* __restrict in,           // [bin][in_stride]
                                                            uint32_t* __restrict tile_sum_out, // [bin][out_stride]
                                                            uint32_t in_stride,                // per-bin stride (elements)
                                                            uint32_t in_real,                  // logical length per bin
                                                            uint32_t out_stride,               // per-bin stride for tile sums
                                                            uint32_t out_real                  // number of valid tiles per bin
                                                  ) {

                                                            // thread geometry
                                                            const uint32_t lane = threadIdx.x; // 0..31
                                                            const uint32_t warp_id = threadIdx.y; // 0..31
                                                            const uint32_t tile = blockIdx.x;  // tile index
                                                            const uint32_t bin = blockIdx.y;

                                                            if (tile >= out_real) return;

                                                            // shared arrays for this CTA
                                                            __shared__ uint32_t warp_sums[BlockSize];   // 32 warp sums
                                                            __shared__ uint32_t warp_prefix[BlockSize]; // exclusive scan of warp sums

                                                            // linear index inside this bin
                                                            const uint32_t linear = tile * (BlockSize * BlockSize) + warp_id * BlockSize + lane;

                                                            // load value (pad with 0 beyond in_real)
                                                            uint32_t value = 0;
                                                            if (linear < in_real) {
                                                                      value = in[bin * in_stride + linear];
                                                            }

                                                            // -----------------------------
                                                            // warp-local exclusive scan for this warp's 32 lanes
                                                            // -----------------------------
                                                            uint32_t warp_ex = 0;
                                                            {
                                                                      uint32_t v = value; // inclusive scan in v, then convert to exclusive
#pragma unroll
                                                                      for (int off = 1; off < 32; off <<= 1) {
                                                                                uint32_t n = __shfl_up_sync(0xffffffffu, v, off);
                                                                                if (lane >= (uint32_t)off) v += n;
                                                                      }
                                                                      uint32_t prev = __shfl_up_sync(0xffffffffu, v, 1);
                                                                      warp_ex = (lane == 0) ? 0u : prev;
                                                            }

                                                            // -----------------------------
                                                            //warp sum (reduce within warp)
                                                            // -----------------------------
                                                            uint32_t sum = value;
#pragma unroll
                                                            for (int off = 16; off > 0; off >>= 1) {
                                                                      sum += __shfl_down_sync(0xffffffffu, sum, off);
                                                            }
                                                            sum = __shfl_sync(0xffffffffu, sum, 0); // broadcast lane0's sum to all lanes

                                                            // lane0 writes the warp sum
                                                            if (lane == 0) {
                                                                      warp_sums[warp_id] = sum;
                                                            }
                                                            __syncthreads();

                                                            // -----------------------------
                                                            // warp0 scans warp_sums[0..31] exclusively -> warp_prefix[warp_id]
                                                            // -----------------------------
                                                            if (warp_id == 0) {
                                                                      uint32_t v = warp_sums[lane]; // lane 0..31
#pragma unroll
                                                                      for (int off = 1; off < 32; off <<= 1) {
                                                                                uint32_t n = __shfl_up_sync(0xffffffffu, v, off);
                                                                                if (lane >= (uint32_t)off) v += n;
                                                                      }
                                                                      uint32_t prev = __shfl_up_sync(0xffffffffu, v, 1);
                                                                      warp_prefix[lane] = (lane == 0) ? 0u : prev; // exclusive
                                                            }
                                                            __syncthreads();

                                                            // -----------------------------
                                                            // write back tile-local exclusive scan
                                                            //     tile-local = warp_prefix[warp_id] + warp_ex
                                                            // -----------------------------
                                                            if (linear < in_real) {
                                                                      in[bin * in_stride + linear] = warp_prefix[warp_id] + warp_ex;
                                                            }

                                                            // -----------------------------
                                                            // emit tile sum
                                                            //     total tile sum = sum of all 1024 elems = warp_prefix[last] + warp_sums[last]
                                                            //     warp_id==0 && lane==0 does it once
                                                            // -----------------------------
                                                            if (warp_id == 0 && lane == 0) {
                                                                      uint32_t tile_sum = warp_prefix[BlockSize - 1] + warp_sums[BlockSize - 1];
                                                                      tile_sum_out[bin * out_stride + tile] = tile_sum;
                                                            }
                                                  }

                                                  // ============================================================================
                                                  // Kernel 2: kernel_exclusive_scan_upto1024_per_bin
                                                  //
                                                  // Role in hierarchy:
                                                  //   * Top-level scan
                                                  //   * Converts TILE SUMS into TILE OFFSETS
                                                  //
                                                  // This kernel is called EXACTLY ONCE per hierarchy,
                                                  // and ONLY for the TOP level.
                                                  //
                                                  // What this kernel DOES:
                                                  //   * Performs an EXCLUSIVE scan over `real <= 1024` elements per bin.
                                                  //   * Overwrites SUMS with EXCLUSIVE OFFSETS.
                                                  //
                                                  // What this kernel DOES NOT DO:
                                                  //   * It does NOT propagate offsets to lower levels.
                                                  //   * It does NOT handle arrays larger than 1024 elements.
                                                  //   * It does NOT touch padding beyond `real`.
                                                  //
                                                  // Why this kernel is special:
                                                  //   * By hierarchy construction, the top level ALWAYS satisfies real <= 1024.
                                                  //   * Therefore, a single CTA per bin is sufficient.
                                                  //
                                                  // Input / Output semantics:
                                                  //   data:
                                                  //     - Input : per-tile SUMS
                                                  //     - Output: per-tile EXCLUSIVE OFFSETS
                                                  //
                                                  // Padding rule:
                                                  //   * Indices >= real are ignored
                                                  //   * Padding must be zero-initialized but is never read semantically
                                                  //
                                                  // Memory layout:
                                                  //   * data : [BinSize][stride]
                                                  //   * stride >= real
                                                  //
                                                  // Launch configuration:
                                                  //   block = dim3(32, 32, 1)   // 1024 threads
                                                  //   grid  = dim3(1, BinSize, 1)
                                                  //
                                                  // Correctness invariants:
                                                  //   * real <= 1024 (guaranteed by hierarchy construction)
                                                  //   * data contains SUMS on entry, OFFSETS on exit
                                                  // ============================================================================

                                                  template<std::size_t BlockSize = 32>
                                                  __global__ void kernel_exclusive_scan_upto1024_per_bin(
                                                            uint32_t* __restrict data,   // [bin][stride]
                                                            uint32_t stride,
                                                            uint32_t real                // logical length (<= 1024)
                                                  ) {
                                                            const uint32_t lane = threadIdx.x; // 0..31
                                                            const uint32_t warp_id = threadIdx.y; // 0..31
                                                            const uint32_t bin = blockIdx.y;

                                                            const uint32_t idx = warp_id * BlockSize + lane; // 0..1023

                                                            // shared
                                                            __shared__ uint32_t warp_sums[BlockSize];
                                                            __shared__ uint32_t warp_prefix[BlockSize];

                                                            uint32_t v = 0;
                                                            if (idx < real) 
                                                                      v = data[bin * stride + idx];

                                                            // warp inclusive -> exclusive
                                                            uint32_t warp_ex = 0;
                                                            {
                                                                      uint32_t x = v;
#pragma unroll
                                                                      for (int off = 1; off < 32; off <<= 1) {
                                                                                uint32_t n = __shfl_up_sync(0xffffffffu, x, off);
                                                                                if (lane >= (uint32_t)off) x += n;
                                                                      }
                                                                      uint32_t prev = __shfl_up_sync(0xffffffffu, x, 1);
                                                                      warp_ex = (lane == 0) ? 0u : prev;
                                                            }

                                                            // warp sum
                                                            uint32_t sum = v;
#pragma unroll
                                                            for (int off = 16; off > 0; off >>= 1) {
                                                                      sum += __shfl_down_sync(0xffffffffu, sum, off);
                                                            }
                                                            sum = __shfl_sync(0xffffffffu, sum, 0);

                                                            if (lane == 0) warp_sums[warp_id] = sum;
                                                            __syncthreads();

                                                            // warp0 scans warp_sums to warp_prefix
                                                            if (warp_id == 0) {
                                                                      uint32_t x = warp_sums[lane];
#pragma unroll
                                                                      for (int off = 1; off < 32; off <<= 1) {
                                                                                uint32_t n = __shfl_up_sync(0xffffffffu, x, off);
                                                                                if (lane >= (uint32_t)off) x += n;
                                                                      }
                                                                      uint32_t prev = __shfl_up_sync(0xffffffffu, x, 1);
                                                                      warp_prefix[lane] = (lane == 0) ? 0u : prev;
                                                            }
                                                            __syncthreads();

                                                            // write exclusive offsets
                                                            if (idx < real) {
                                                                      data[bin * stride + idx] = warp_prefix[warp_id] + warp_ex;
                                                            }
                                                  }

                                                  template<std::size_t BlockSize = 32>
                                                  __device__ __forceinline__ void apply_tile_offset_1024(
                                                                      uint32_t* __restrict dst,        // [bin][dst_stride]
                                                                      const uint32_t* __restrict src,  // [bin][src_stride]
                                                                      uint32_t dst_stride,
                                                                      uint32_t dst_real,
                                                                      uint32_t src_stride,
                                                                      uint32_t src_real
                                                            ) {
                                                            const uint32_t lane = threadIdx.x;
                                                            const uint32_t warp_id = threadIdx.y;
                                                            const uint32_t tile = blockIdx.x;
                                                            const uint32_t bin = blockIdx.y;

                                                            const uint32_t linear =
                                                                      tile * (BlockSize * BlockSize) +
                                                                      warp_id * BlockSize +
                                                                      lane;

                                                            if (linear >= dst_real) return;

                                                            uint32_t v = dst[bin * dst_stride + linear];

                                                            const uint32_t t = linear >> 10; // /1024
                                                            if (t < src_real) {
                                                                      v += src[bin * src_stride + t];
                                                            }

                                                            dst[bin * dst_stride + linear] = v;
                                                  }


                                                  // ================================================================
                                                  // Kernel 3: kernel_apply_tile_offsets_to_element_offsets_1024
                                                  //
                                                  // Purpose:
                                                  //   Propagate tile-level OFFSETS down to element-level OFFSETS.
                                                  //
                                                  // Operation:
                                                  //   For each element:
                                                  //       dst_offsets[bin][i] += src_offsets[bin][ i >> 10 ]
                                                  //
                                                  // Where:
                                                  //   - dst_offsets is element-level data (length = dst_real)
                                                  //   - src_offsets is tile-level offsets
                                                  //   - i >> 10 == i / 1024 selects the tile index
                                                  //
                                                  // Typical usage:
                                                  //   - Downsweep phase in a multi-level hierarchy
                                                  //   - Applies higher-level offsets to lower-level arrays
                                                  //
                                                  // Memory layout:
                                                  //   - dst_offsets: [BinSize][dst_stride]
                                                  //   - src_offsets: [BinSize][src_stride]
                                                  //
                                                  // Launch configuration:
                                                  //   block = dim3(32, 32, 1)
                                                  //   grid  = dim3(ceil(dst_real / 1024), BinSize, 1)
                                                  //
                                                  // Correctness invariants:
                                                  //   - src_real == ceil(dst_real / 1024)
                                                  //   - dst_offsets already contains EXCLUSIVE values locally
                                                  // ================================================================
                                                  template<std::size_t BlockSize = 32>
                                                  __global__ void kernel_apply_tile_offsets_to_element_offsets_1024(
                                                            uint32_t* __restrict dst_offsets,       // level[k-1]
                                                            const uint32_t* __restrict src_offsets, // level[k]
                                                            uint32_t dst_stride, uint32_t dst_real,
                                                            uint32_t src_stride, uint32_t src_real
                                                  ) {
                                                            apply_tile_offset_1024<BlockSize>(
                                                                      dst_offsets,
                                                                      src_offsets,
                                                                      dst_stride,
                                                                      dst_real,
                                                                      src_stride,
                                                                      src_real
                                                            );
                                                  }

                                                  // ============================================================================
                                                  // Kernel 4: kernel_apply_level0_to_localT_1024
                                                  //
                                                  // Role in hierarchy:
                                                  //   * Final assembly step
                                                  //   * Produces the FULL EXCLUSIVE SCAN over the original input
                                                  //
                                                  // Conceptually:
                                                  //   localT[i] += level0[i / 1024]
                                                  //
                                                  // What this kernel DOES:
                                                  //   * Combines:
                                                  //       - tile-local EXCLUSIVE scans stored in localT
                                                  //       - tile-level EXCLUSIVE OFFSETS stored in level0
                                                  //   * Produces the final global EXCLUSIVE scan result
                                                  //
                                                  // What this kernel DOES NOT DO:
                                                  //   * It does NOT perform any scan.
                                                  //   * It does NOT modify level0.
                                                  //
                                                  // Preconditions:
                                                  //   * localT contains tile-local EXCLUSIVE scan results
                                                  //   * level0 contains tile-level EXCLUSIVE OFFSETS
                                                  //   * level0_real == ceil(local_real / 1024)
                                                  //
                                                  // Memory layout:
                                                  //   * localT : [BinSize][local_stride]
                                                  //   * level0 : [BinSize][level0_stride]
                                                  //
                                                  // Launch configuration:
                                                  //   block = dim3(32, 32, 1)
                                                  //   grid  = dim3(ceil(local_real / 1024), BinSize, 1)
                                                  //
                                                  // Correctness invariants:
                                                  //   * Every element receives exactly one tile offset
                                                  //   * Final result is a correct EXCLUSIVE scan over localT
                                                  // ============================================================================
                                                  template<std::size_t BlockSize = 32>
                                                  __global__ void kernel_apply_level0_to_localT_1024(
                                                            uint32_t* __restrict localT,
                                                            const uint32_t* __restrict level0,
                                                            uint32_t local_stride, uint32_t local_real,
                                                            uint32_t level0_stride, uint32_t level0_real
                                                  ) {
                                                            apply_tile_offset_1024<BlockSize>(
                                                                      localT,
                                                                      level0,
                                                                      local_stride,
                                                                      local_real,
                                                                      level0_stride,
                                                                      level0_real
                                                            );
                                                  }

                                        } // namespace v1
                              } // namespace details
                    } // namespace radix
          } // namespace gpu
} // namespace sort

#endif //_CUDA_HIERARCHICAL_EXCLUSIVE_SCAN_CUH_
