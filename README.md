# libHPC

libHPC is a high-performance computing library focused on Linux and Windows
environments. It provides SIMD-optimized kernels, concurrent data structures,
GPU utilities, and HPC-oriented memory management components.



## 0x00 Platform Support

| Platform                          | Status               |
| --------------------------------- | -------------------- |
| **Linux (x86_64 / CUDA)**         | ✓ Supported          |
| **Windows (MSVC / CUDA)**         | ✓ Supported          |
| **macOS (Intel)**                 | ✓ Supported, limited |
| **macOS (Apple Silicon / ARM64)** | ✗ **Not supported**  |



## 0x01 macOS Apple Silicon Notice

libHPC **does not support macOS ARM (Apple Silicon)**.

The reason is simple:

> **Apple’s recent macOS / Xcode toolchain updates introduced ABI changes in `libc++`, causing oneTBB and other HPC components to fail at link-time.**  
>
> These issues do not occur on Linux or Windows, and they did not occur on older macOS versions.

Since the goal of libHPC is stable, reproducible high-performance computing,
macOS ARM is excluded to avoid degraded reliability or performance.

If the Apple toolchain becomes ABI-stable again or TBB provides upstream fixes,
support may be reconsidered in the future.

## 0x02 macOS ARM Support

libHPC previously worked on macOS ARM.  
However, new Xcode toolchains explicitly mark several `libc++` ABI symbols as
**forbidden** (Xcode even shows a “prohibited symbol” icon), including
`std::__1::__hash_memory`, which oneTBB depends on.

Since these symbols are removed at the SDK level, the issue cannot be fixed in
libHPC or by configuration changes. As a result, macOS ARM support has been
formally dropped.
