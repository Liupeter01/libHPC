#include <single_mempool_impl.h>
#include <string.h>

static void *ngx_palloc_small(struct ngx_pool_s *pool, size_t size,
                              unsigned char is_align);
static void *ngx_palloc_block(struct ngx_pool_s *pool, size_t size);
static void *ngx_palloc_large(struct ngx_pool_s *pool, size_t size);

static void *aligned_alloc_portable(size_t alignment, size_t size) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#elif defined(__APPLE__) || defined(__linux__)
  void *ptr = NULL;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return NULL;
  }
  return ptr;
#else
#error "Platform not supported"
#endif
}

void aligned_free_portable(void *ptr) {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

struct ngx_pool_s *ngx_create_pool(const size_t size) {
  struct ngx_pool_s *p = aligned_alloc_portable(NGX_POOL_ALIGNMENT, size);
  if (p == NULL)
    return NULL;

  p->d.last = (unsigned char *)p + sizeof(struct ngx_pool_s);
  p->d.end = (unsigned char *)p + size;
  p->d.next = NULL;
  p->d.failed = 0;

  p->max = p->d.end - p->d.last < NGX_MAX_ALLOC_FROM_POOL
               ? p->d.end - p->d.last
               : NGX_MAX_ALLOC_FROM_POOL;

  p->current = p;
  p->large = NULL;
  p->cleanup = NULL;
  return p;
}

void ngx_destroy_pool(struct ngx_pool_s *pool) {
  struct ngx_pool_s *p, *n;
  struct ngx_pool_large_s *l, *m;
  struct ngx_pool_cleanup_s *c;

  for (c = pool->cleanup; c; c = c->next) {
    if (c->handler) {
      c->handler(c->data);
    }
  }

  for (l = pool->large; l; l = l->next) {
    if (l->alloc) {
      free(l->alloc);
    }
  }

  for (p = pool, n = pool->d.next; /* void */; p = n, n = n->d.next) {
    aligned_free_portable(p);
    if (n == NULL)
      break;
  }
}

void ngx_reset_pool(struct ngx_pool_s *pool) {
  struct ngx_pool_s *p;
  struct ngx_pool_large_s *l;

  for (l = pool->large; l; l = l->next) {
    if (l->alloc) {
      aligned_free_portable(l->alloc);
    }
  }

  for (p = pool; p; p = p->d.next) {
    if (p == pool)
      p->d.last = (unsigned char *)p + sizeof(struct ngx_pool_s);
    else
      p->d.last = (unsigned char *)p + sizeof(ngx_pool_data_t);
    p->d.failed = 0;
  }

  pool->current = pool;
  pool->large = NULL;
}

void *ngx_palloc(struct ngx_pool_s *pool, size_t size) {
  if (!size)
    return NULL;

  if (size <= pool->max) {
    return ngx_palloc_small(pool, size, 1);
  }
  return ngx_palloc_large(pool, size);
}

void *ngx_pnalloc(struct ngx_pool_s *pool, size_t size) {
  if (!size)
    return NULL;

  if (size <= pool->max) {
    return ngx_palloc_small(pool, size, 0);
  }
  return ngx_palloc_large(pool, size);
}

static void *ngx_palloc_small(struct ngx_pool_s *pool, size_t size,
                              unsigned char is_align) {
  if (!size)
    return NULL;

  struct ngx_pool_s *p = pool->current;

  do {
    unsigned char *m = p->d.last;

    if (is_align) {
      m = ngx_align_ptr(m, NGX_ALIGNMENT);
    }

    if ((size_t)(p->d.end - m) >= size) {
      p->d.last = m + size;
      return m;
    }

    p = p->d.next;

  } while (p);

  return ngx_palloc_block(pool, size);
}

static void *ngx_palloc_block(struct ngx_pool_s *pool, size_t size) {
  if (!size)
    return NULL;

  struct ngx_pool_s *p;
  const size_t psize = (size_t)(pool->d.end - (unsigned char *)pool);
  unsigned char *m = aligned_alloc_portable(NGX_POOL_ALIGNMENT, psize);
  if (m == NULL)
    return NULL;

  struct ngx_pool_s *new = (struct ngx_pool_s *)m;

  new->d.end = m + psize;
  new->d.next = NULL;
  new->d.failed = 0;

  m += sizeof(ngx_pool_data_t);
  m = ngx_align_ptr(m, NGX_ALIGNMENT);
  new->d.last = m + size;

  for (p = pool->current; p->d.next; p = p->d.next) {
    if (p->d.failed++ > 4) {
      pool->current = p->d.next;
    }
  }

  p->d.next = new;
  return m;
}

static void *ngx_palloc_large(struct ngx_pool_s *pool, size_t size) {
  if (!size)
    return NULL;

  void *p = malloc(size); // because it's large memory
  if (p == NULL)
    return NULL;

  uintptr_t n = 0;
  struct ngx_pool_large_s *large;
  for (large = pool->large; large; large = large->next) {
    if (large->alloc == NULL) {
      large->alloc = p;
      return p;
    }

    if (n++ > 3) {
      break;
    }
  }

  large = ngx_palloc_small(pool, sizeof(struct ngx_pool_large_s), 1);
  if (large == NULL) {
    free(p); // it could be removed by free!
    return NULL;
  }

  large->alloc = p;
  large->next = pool->large;
  pool->large = large;
  return p;
}

int ngx_pfree(struct ngx_pool_s *pool, void *p) {
  struct ngx_pool_large_s *l;

  if (p == NULL) {
    return -1;
  }

  for (l = pool->large; l; l = l->next) {
    if (p == l->alloc) {
      free(l->alloc);
      l->alloc = NULL;
      return 0;
    }
  }

  return -1;
}

void *ngx_pcalloc(struct ngx_pool_s *pool, size_t size) {
  void *p = ngx_palloc(pool, size);
  if (p) {
    memset(p, 0, size);
  }
  return p;
}

struct ngx_pool_cleanup_s *ngx_pool_cleanup_add(struct ngx_pool_s *p,
                                                size_t size) {
  struct ngx_pool_cleanup_s *c =
      ngx_palloc(p, sizeof(struct ngx_pool_cleanup_s));
  if (c == NULL)
    return NULL;

  if (size) {
    c->data = ngx_palloc(p, size);
    if (c->data == NULL) {
      return NULL;
    }

  } else {
    c->data = NULL;
  }

  c->handler = NULL;
  c->next = p->cleanup;

  p->cleanup = c;
  return c;
}
