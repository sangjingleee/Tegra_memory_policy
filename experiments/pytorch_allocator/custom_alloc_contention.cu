#include <cuda_runtime.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define POOL_MAX 16384
#define ZC_POOL_MAX 8192
#define MANAGED_POOL_MAX 8192

static size_t s_policy_threshold = 4ULL * 1024 * 1024;
static volatile int s_phase = 0;
static int s_strategy = -1;

#define LOG_KINDS 4
#define LOG_BUCKETS 8

static const char* s_kind_names[LOG_KINDS] = {
    "device", "managed", "mapped_zc", "device_fallback"
};

static const size_t s_bucket_max[LOG_BUCKETS] = {
    64ULL * 1024,
    256ULL * 1024,
    1024ULL * 1024,
    4ULL * 1024 * 1024,
    16ULL * 1024 * 1024,
    64ULL * 1024 * 1024,
    256ULL * 1024 * 1024,
    (size_t)-1
};

static const char* s_bucket_names[LOG_BUCKETS] = {
    "<64KB", "64KB-256KB", "256KB-1MB", "1MB-4MB",
    "4MB-16MB", "16MB-64MB", "64MB-256MB", ">=256MB"
};

static unsigned long long s_log_count[LOG_KINDS][LOG_BUCKETS];
static unsigned long long s_log_bytes[LOG_KINDS][LOG_BUCKETS];
static pthread_mutex_t s_log_mtx = PTHREAD_MUTEX_INITIALIZER;

typedef struct PoolEntry {
    void* ptr;
    size_t size;
    int in_use;
} PoolEntry;

static PoolEntry s_dev_pool[POOL_MAX];
static int s_dev_count = 0;
static pthread_mutex_t s_dev_mtx = PTHREAD_MUTEX_INITIALIZER;

static PoolEntry s_mgd_pool[MANAGED_POOL_MAX];
static int s_mgd_count = 0;
static pthread_mutex_t s_mgd_mtx = PTHREAD_MUTEX_INITIALIZER;

typedef struct ZCEntry {
    void* host_ptr;
    void* dev_ptr;
    size_t size;
    int in_use;
} ZCEntry;

static ZCEntry s_zc_pool[ZC_POOL_MAX];
static int s_zc_count = 0;
static pthread_mutex_t s_zc_mtx = PTHREAD_MUTEX_INITIALIZER;

static int get_strategy(void) {
    if (s_strategy < 0) {
        const char* e = getenv("ALLOC_STRATEGY");
        s_strategy = e ? atoi(e) : 0;
    }
    return s_strategy;
}

static int bucket_for_size(size_t size) {
    for (int i = 0; i < LOG_BUCKETS; i++) {
        if (size < s_bucket_max[i]) return i;
    }
    return LOG_BUCKETS - 1;
}

static void record_alloc(int kind, size_t size) {
    if (s_phase != 1) return;
    if (kind < 0 || kind >= LOG_KINDS) return;
    int b = bucket_for_size(size);
    pthread_mutex_lock(&s_log_mtx);
    s_log_count[kind][b] += 1;
    s_log_bytes[kind][b] += (unsigned long long)size;
    pthread_mutex_unlock(&s_log_mtx);
}

static PoolEntry* pool_find_free(PoolEntry* pool, int count, size_t size) {
    for (int i = 0; i < count; i++) {
        if (!pool[i].in_use && pool[i].size == size) return &pool[i];
    }
    return NULL;
}

static PoolEntry* pool_find_ptr(PoolEntry* pool, int count, void* ptr) {
    for (int i = 0; i < count; i++) {
        if (pool[i].ptr == ptr) return &pool[i];
    }
    return NULL;
}

static void* dev_alloc(size_t size) {
    pthread_mutex_lock(&s_dev_mtx);
    PoolEntry* e = pool_find_free(s_dev_pool, s_dev_count, size);
    if (e) {
        e->in_use = 1;
        void* p = e->ptr;
        pthread_mutex_unlock(&s_dev_mtx);
        return p;
    }
    if (s_dev_count < POOL_MAX) {
        void* ptr = NULL;
        cudaError_t err = cudaMalloc(&ptr, size);
        if (err != cudaSuccess) {
            pthread_mutex_unlock(&s_dev_mtx);
            fprintf(stderr, "[contention_alloc] cudaMalloc failed size=%zu err=%d\n", size, (int)err);
            return NULL;
        }
        s_dev_pool[s_dev_count++] = {ptr, size, 1};
        pthread_mutex_unlock(&s_dev_mtx);
        return ptr;
    }
    pthread_mutex_unlock(&s_dev_mtx);
    void* ptr = NULL;
    cudaMalloc(&ptr, size);
    return ptr;
}

static int dev_free(void* ptr) {
    pthread_mutex_lock(&s_dev_mtx);
    PoolEntry* e = pool_find_ptr(s_dev_pool, s_dev_count, ptr);
    if (e) {
        e->in_use = 0;
        pthread_mutex_unlock(&s_dev_mtx);
        return 1;
    }
    pthread_mutex_unlock(&s_dev_mtx);
    return 0;
}

static void* managed_alloc(size_t size) {
    pthread_mutex_lock(&s_mgd_mtx);
    PoolEntry* e = pool_find_free(s_mgd_pool, s_mgd_count, size);
    if (e) {
        e->in_use = 1;
        void* p = e->ptr;
        pthread_mutex_unlock(&s_mgd_mtx);
        return p;
    }
    if (s_mgd_count < MANAGED_POOL_MAX) {
        void* ptr = NULL;
        cudaError_t err = cudaMallocManaged(&ptr, size, cudaMemAttachGlobal);
        if (err != cudaSuccess) {
            pthread_mutex_unlock(&s_mgd_mtx);
            fprintf(stderr, "[contention_alloc] cudaMallocManaged failed size=%zu err=%d\n", size, (int)err);
            return NULL;
        }
        s_mgd_pool[s_mgd_count++] = {ptr, size, 1};
        pthread_mutex_unlock(&s_mgd_mtx);
        return ptr;
    }
    pthread_mutex_unlock(&s_mgd_mtx);
    return dev_alloc(size);
}

static int managed_free(void* ptr) {
    pthread_mutex_lock(&s_mgd_mtx);
    PoolEntry* e = pool_find_ptr(s_mgd_pool, s_mgd_count, ptr);
    if (e) {
        e->in_use = 0;
        pthread_mutex_unlock(&s_mgd_mtx);
        return 1;
    }
    pthread_mutex_unlock(&s_mgd_mtx);
    return 0;
}

static ZCEntry* zc_find_free(size_t size) {
    for (int i = 0; i < s_zc_count; i++) {
        if (!s_zc_pool[i].in_use && s_zc_pool[i].size == size) return &s_zc_pool[i];
    }
    return NULL;
}

static ZCEntry* zc_find_ptr(void* ptr) {
    for (int i = 0; i < s_zc_count; i++) {
        if (s_zc_pool[i].dev_ptr == ptr) return &s_zc_pool[i];
    }
    return NULL;
}

static void* zc_alloc(size_t size) {
    if (size == 0) return NULL;
    pthread_mutex_lock(&s_zc_mtx);
    ZCEntry* e = zc_find_free(size);
    if (e) {
        e->in_use = 1;
        void* p = e->dev_ptr;
        pthread_mutex_unlock(&s_zc_mtx);
        return p;
    }
    if (s_zc_count < ZC_POOL_MAX) {
        void *host_ptr = NULL, *dev_ptr = NULL;
        cudaError_t err = cudaHostAlloc(&host_ptr, size, cudaHostAllocMapped | cudaHostAllocPortable);
        if (err == cudaSuccess) err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
        if (err != cudaSuccess) {
            if (host_ptr) cudaFreeHost(host_ptr);
            pthread_mutex_unlock(&s_zc_mtx);
            fprintf(stderr, "[contention_alloc] zc alloc failed size=%zu err=%d\n", size, (int)err);
            return NULL;
        }
        s_zc_pool[s_zc_count++] = {host_ptr, dev_ptr, size, 1};
        pthread_mutex_unlock(&s_zc_mtx);
        return dev_ptr;
    }
    pthread_mutex_unlock(&s_zc_mtx);
    return dev_alloc(size);
}

static int zc_free(void* ptr) {
    pthread_mutex_lock(&s_zc_mtx);
    ZCEntry* e = zc_find_ptr(ptr);
    if (e) {
        e->in_use = 0;
        pthread_mutex_unlock(&s_zc_mtx);
        return 1;
    }
    pthread_mutex_unlock(&s_zc_mtx);
    return 0;
}

extern "C" {

void set_phase(int p) { s_phase = p; }
int get_phase(void) { return s_phase; }
void set_threshold(size_t t) { s_policy_threshold = t; }
size_t get_threshold(void) { return s_policy_threshold; }

void* custom_malloc(size_t size, int device, cudaStream_t stream) {
    if (size == 0) return NULL;
    int strat = (s_phase == 1) ? get_strategy() : 0;
    if (strat == 1) {
        void* p = zc_alloc(size);
        if (p) {
            record_alloc(2, size);
            return p;
        }
        record_alloc(3, size);
        return dev_alloc(size);
    }
    if (strat == 2 && size >= s_policy_threshold) {
        void* p = zc_alloc(size);
        if (p) {
            record_alloc(2, size);
            return p;
        }
        record_alloc(3, size);
        return dev_alloc(size);
    }
    if (strat == 4 && size >= s_policy_threshold) {
        void* p = managed_alloc(size);
        if (p) {
            record_alloc(1, size);
            return p;
        }
        record_alloc(3, size);
        return dev_alloc(size);
    }
    record_alloc(0, size);
    return dev_alloc(size);
}

void custom_free(void* ptr, size_t size, int device, cudaStream_t stream) {
    if (zc_free(ptr)) return;
    if (managed_free(ptr)) return;
    if (dev_free(ptr)) return;
    cudaFree(ptr);
}

void dump_alloc_summary(const char* path) {
    if (!path || !path[0]) return;
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "kind,bucket,count,total_bytes,total_mb\n");
    pthread_mutex_lock(&s_log_mtx);
    for (int k = 0; k < LOG_KINDS; k++) {
        for (int b = 0; b < LOG_BUCKETS; b++) {
            unsigned long long count = s_log_count[k][b];
            unsigned long long bytes = s_log_bytes[k][b];
            if (count == 0) continue;
            fprintf(
                f,
                "%s,%s,%llu,%llu,%.6f\n",
                s_kind_names[k],
                s_bucket_names[b],
                count,
                bytes,
                (double)bytes / 1048576.0
            );
        }
    }
    pthread_mutex_unlock(&s_log_mtx);
    fclose(f);
}

}
