/*
 * single_mempool_impl.h - A lightweight single-threaded memory pool
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
#ifndef _SINGLE_MEMPOOL_IMPL_H_
#define _SINGLE_MEMPOOL_IMPL_H_
#include <def/config.h>
#include <def/ds.h>

struct ngx_pool_s *ngx_create_pool(const size_t size);
void ngx_destroy_pool(struct ngx_pool_s *pool);
void ngx_reset_pool(struct ngx_pool_s *pool);

void *ngx_palloc(struct ngx_pool_s *pool, size_t size);
void *ngx_pnalloc(struct ngx_pool_s *pool, size_t size);
void *ngx_pcalloc(struct ngx_pool_s *pool, size_t size);
int ngx_pfree(struct ngx_pool_s *pool, void *p);

struct ngx_pool_cleanup_s *ngx_pool_cleanup_add(struct ngx_pool_s *p,
                                                size_t size);

#endif //_SINGLE_MEMPOOL_IMPL_H_
