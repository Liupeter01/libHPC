/*
 * config.h - A lightweight single-threaded memory pool
 *
 * This implementation is inspired by the design of ngx_palloc.c from the nginx
 * project.
 *
 * Copyright (c) 2025 Jonathan Liu
 * Portions derived from Nginx (https://nginx.org), which is licensed under the
 * BSD-2-Clause License.
 *
 * This file is part of a larger project licensed under the MIT License.
 * See LICENSE for details.
 */

#pragma once
#ifndef _CONFIG_H_
#define _CONFIG_H_

#define NGX_POOL_ALIGNMENT 16
#define PAGE_SIZE (1 << 12)
#define ngx_align(d, a) (((d) + (a - 1)) & ~(a - 1))
#define ngx_align_ptr(p, a)                                                    \
  (unsigned char *)(((uintptr_t)(p) + ((uintptr_t)a - 1)) & ~((uintptr_t)a - 1))

/*
 * NGX_MAX_ALLOC_FROM_POOL should be (ngx_pagesize - 1), i.e. 4095 on x86.
 * On Windows NT it decreases a number of locked pages in a kernel.
 */
#define NGX_MAX_ALLOC_FROM_POOL (PAGE_SIZE - 1)
#define NGX_DEFAULT_POOL_SIZE (16 * 1024)

/*def pointer size!*/
#ifndef NGX_ALIGNMENT
#define NGX_ALIGNMENT sizeof(uintptr_t) /* platform word */
#endif

#endif //_CONFIG_H_
