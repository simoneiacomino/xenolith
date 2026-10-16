#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "kvstore.h"
#include "format.h"
#include "xenolith_internal.h"
#include <sys/file.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    KVSTORE_META_SIZE = 128,
    KVSTORE_ANCHOR_SIZE = 64,
    KVSTORE_META_KEY = 1,
    KVSTORE_META_GROUP = 2
};

typedef struct {
    uint8_t hash[32];
    uint64_t length;
} kvstore_shared_ref;

typedef struct {
    uint64_t size, prefix, data_offset;
    kvstore_id group;
    size_t count;
    kvstore_shared_ref *refs;
} kvstore_shared_map;

typedef struct kvstore_shared_plan kvstore_shared_plan;

typedef struct {
    kvstore_id id;
    uint8_t key[32];
    uint64_t payload_size;
    uint64_t hits;
    int64_t last_hit;
    uint64_t rebuild_cost;
    int64_t created;
    uint32_t flags;
    kvstore_id group;
} kvstore_meta;

typedef struct {
    kvstore_meta meta;
    char snapshot_name[37];
    char meta_name[38];
    int pinned;
    int invalid;
    uint64_t bytes;
    int64_t used;
    kvstore_shared_map map;
} kvstore_candidate;

struct kvstore {
    int directory_fd;
    uint64_t budget;
    int lock_fd;
    struct { kvstore_id id, group; } *groups;
    size_t group_count;
    const kvstore_id *protect_group, *protect_id;
};

__attribute__((weak)) int kvstore_fault(const char *point) {
    (void)point;
    return 0;
}

enum {
    KVSTORE_SHARED_HEADER_SIZE = 96,
    KVSTORE_SHARED_REF_SIZE = 40,
    KVSTORE_CHUNK_SIZE = 256 * 1024,
    KVSTORE_GLOBAL_ROW_SIZE = 1024,
    KVSTORE_GLOBAL_HEADS = 20,
    KVSTORE_SHARED_MAX_REFS = KVSTORE_GLOBAL_HEADS * 1024
};
/* The envelope contains an independent map followed by the unchanged snapshot
 * header, tokens, logits and SWA. Global head streams are split on fixed row
 * boundaries and addressed by SHA-256. No checkpoint depends on another map.
 * Only complete, fsynced chunks are referenced by a published manifest. */

struct kvstore_shared_plan {
    int dir, output, writing;
    uint64_t position, head_bytes, reserve;
    size_t next, buffered;
    uint8_t *buffer, *missing, *created;
    kvstore_shared_map map;
};

static int kvstore_io_at(int fd, void *data, size_t n, uint64_t off, int writing) {
    uint8_t *p = data;
    while (n) {
        ssize_t got = writing ? pwrite(fd, p, n, (off_t)off) : pread(fd, p, n, (off_t)off);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return 0;
        p += got; n -= (size_t)got; off += (uint64_t)got;
    }
    return 1;
}
static void kvstore_shared_name(const uint8_t hash[32], char out[69]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[2*i] = hex[hash[i] >> 4]; out[2*i+1] = hex[hash[i] & 15]; }
    memcpy(out + 64, ".xkc", 5);
}
static int kvstore_shared_parse_name(const char *name, uint8_t hash[32]) {
    static const char hex[] = "0123456789abcdef";
    if (strlen(name) != 68 || strcmp(name + 64, ".xkc")) return 0;
    for (int i = 0; i < 32; i++) {
        const char *a = strchr(hex, name[2*i]);
        const char *b = strchr(hex, name[2*i+1]);
        if (!a || !b) return 0;
        hash[i] = (uint8_t)((a - hex) * 16 + (b - hex));
    }
    return 1;
}
static void kvstore_shared_encode_ref(const kvstore_shared_ref *ref, uint8_t out[KVSTORE_SHARED_REF_SIZE]) {
    memcpy(out, ref->hash, 32); format_put_u64le(out + 32, ref->length);
}
/* 1 shared, 0 standalone, -1 invalid/I/O. Caller frees map.refs. */
static int kvstore_shared_inspect(int fd, kvstore_shared_map *m) {
    memset(m, 0, sizeof *m);
    uint8_t h[KVSTORE_SHARED_HEADER_SIZE]; struct stat st;
    if (!kvstore_io_at(fd, h, 8, 0, 0)) return -1;
    if (memcmp(h, "XEKVSEG\n", 8)) return 0;
    if (!kvstore_io_at(fd, h, sizeof h, 0, 0) || fstat(fd, &st) || !S_ISREG(st.st_mode)) return -1;
    if (format_get_u32le(h + 8) != 1 || format_get_u32le(h + 12) != KVSTORE_SHARED_HEADER_SIZE ||
        format_get_u64le(h + 88) != format_crc64(0, h, 88)) return -1;
    for (int i = 56; i < 80; i++) if (h[i]) return -1;
    m->size = format_get_u64le(h + 16); m->prefix = format_get_u64le(h + 24);
    memcpy(m->group.bytes, h + 32, 16);
    uint64_t count = format_get_u64le(h + 48);
    if (!count || count > KVSTORE_SHARED_MAX_REFS || m->prefix < 640 || m->size > INT64_MAX ||
        m->prefix >= m->size) return -1;
    m->data_offset = KVSTORE_SHARED_HEADER_SIZE + count * KVSTORE_SHARED_REF_SIZE;
    if (st.st_size < 0 || m->data_offset > (uint64_t)st.st_size ||
        m->prefix != (uint64_t)st.st_size - m->data_offset) return -1;
    m->refs = calloc((size_t)count, sizeof *m->refs);
    if (!m->refs) return -1;
    m->count = (size_t)count;
    uint64_t size = m->prefix, crc = 0;
    for (size_t i = 0; i < m->count; i++) {
        uint8_t r[KVSTORE_SHARED_REF_SIZE];
        if (!kvstore_io_at(fd, r, sizeof r, KVSTORE_SHARED_HEADER_SIZE + i * KVSTORE_SHARED_REF_SIZE, 0)) goto bad;
        crc = format_crc64(crc, r, sizeof r);
        memcpy(m->refs[i].hash, r, 32);
        uint64_t n = m->refs[i].length = format_get_u64le(r + 32);
        if (!n || n > KVSTORE_CHUNK_SIZE || n > m->size - size) goto bad;
        size += n;
    }
    if (size != m->size || crc != format_get_u64le(h + 80)) goto bad;
    return 1;
bad:
    free(m->refs); memset(m, 0, sizeof *m); return -1;
}

static int kvstore_shared_complete_chunk(kvstore_shared_plan *p) {
    if (p->next >= p->map.count) return 0;
    kvstore_shared_ref *r = &p->map.refs[p->next];
    uint8_t digest[32]; format_sha256 hash;
    format_sha256_init(&hash); format_sha256_update(&hash, p->buffer, p->buffered);
    format_sha256_final(&hash, digest);
    if (!p->writing) {
        memcpy(r->hash, digest, 32); r->length = p->buffered;
        char name[69]; kvstore_shared_name(digest, name);
        int fd = openat(p->dir, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        if (fd < 0 && errno != ENOENT) return 0;
        if (fd >= 0) {
            struct stat st;
            uint8_t *saved = malloc(p->buffered);
            if (!saved || fstat(fd, &st) || !S_ISREG(st.st_mode)) {
                free(saved);
                close(fd);
                return 0;
            }
            int matches = st.st_size == (off_t)p->buffered &&
                          kvstore_io_at(fd, saved, p->buffered, 0, 0) &&
                          !memcmp(saved, p->buffer, p->buffered);
            free(saved);
            close(fd);
            /* A recomputed chunk can repair a corrupt shared file. Reserve its
             * entire temporary replacement before touching the old file. */
            if (!matches) p->missing[p->next] = 2;
        } else {
            p->missing[p->next] = 1;
        }
    } else {
        if (r->length != p->buffered || memcmp(r->hash, digest, 32)) return 0;
        if (p->missing[p->next]) {
            char name[69], temporary[80];
            kvstore_shared_name(digest, name);
            snprintf(temporary, sizeof temporary, ".tmp-%s", name);
            int fd = openat(p->dir, temporary,
                            O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
            if (fd < 0) return 0;
            int ok = kvstore_io_at(fd, p->buffer, p->buffered, 0, 1) &&
                     !kvstore_fault("chunk-written") && fsync(fd) == 0;
            if (close(fd)) ok = 0;
            if (!ok) return 0;
            unsigned flags = p->missing[p->next] == 1 ? RENAME_NOREPLACE : 0;
            if (renameat2(p->dir, temporary, p->dir, name, flags)) return 0;
            p->created[p->next] = p->missing[p->next] == 1;
            /* Repeated identical chunks need no second write or repair. */
            for (size_t i = p->next + 1; i < p->map.count; i++)
                if (!memcmp(p->map.refs[i].hash, digest, 32)) p->missing[i] = 0;
        }
    }
    p->next++; p->buffered = 0; return 1;
}
static ssize_t kvstore_shared_sink_write(void *opaque, const char *data, size_t length) {
    kvstore_shared_plan *p = opaque; size_t original = length;
    while (length) {
        if (p->position < p->map.prefix) {
            uint64_t available = p->map.prefix - p->position;
            size_t take = length < available ? length : (size_t)available;
            if (p->writing && !kvstore_io_at(p->output, (void *)data, take,
                                     p->map.data_offset + p->position, 1)) return -1;
            p->position += take; data += take; length -= take;
        } else {
            if (p->position >= p->map.size) return -1;
            uint64_t within = (p->position - p->map.prefix) % p->head_bytes;
            size_t take = KVSTORE_CHUNK_SIZE - p->buffered;
            if (take > p->head_bytes - within) take = (size_t)(p->head_bytes - within);
            if (take > length) take = length;
            memcpy(p->buffer + p->buffered, data, take);
            p->buffered += take; p->position += take; data += take; length -= take;
            if ((p->buffered == KVSTORE_CHUNK_SIZE || (p->position - p->map.prefix) % p->head_bytes == 0) &&
                !kvstore_shared_complete_chunk(p)) return -1;
        }
    }
    return (ssize_t)original;
}
static int kvstore_shared_sink_seek(void *opaque, off64_t *off, int whence) {
    kvstore_shared_plan *p = opaque;
    if (whence != SEEK_SET || *off < 0 || (uint64_t)*off > p->map.size) return -1;
    /* Global serialization is sequential. Rewrites are confined to header/SWA. */
    if ((uint64_t)*off >= p->map.prefix && (uint64_t)*off != p->position) return -1;
    p->position = (uint64_t)*off; return 0;
}
static xe_snapshot_status kvstore_shared_stream(kvstore_shared_plan *p, xe_session *s) {
    cookie_io_functions_t io = {.write = kvstore_shared_sink_write, .seek = kvstore_shared_sink_seek};
    FILE *sink = fopencookie(p, "wb", io);
    if (!sink) return XE_SNAPSHOT_IO;
    xe_snapshot_status status = xe_session_snapshot_stream(s, sink);
    if (fclose(sink) || p->next != p->map.count || p->buffered) status = XE_SNAPSHOT_IO;
#ifdef XE_TEST_CACHE_TRACE
    if (status != XE_SNAPSHOT_OK)
        fprintf(stderr, "cache stream: writing=%d pos=%llu refs=%zu/%zu buffered=%zu status=%s errno=%d\n",
                p->writing, (unsigned long long)p->position, p->next, p->map.count,
                p->buffered, xe_snapshot_status_name(status), errno);
#endif
    return status;
}
static int kvstore_shared_ref_compare(const void *a, const void *b) {
    return memcmp(((const kvstore_shared_ref *)a)->hash, ((const kvstore_shared_ref *)b)->hash, 32);
}
static void kvstore_shared_finish(kvstore_shared_plan *p, int committed);

/* Preparation only measures/reuses chunks; the store lock covers the plan. */
static xe_snapshot_status kvstore_shared_prepare(int dir, xe_session *s, const kvstore_id *group,
                                   kvstore_shared_plan **out) {
    *out = NULL; uint64_t size;
    xe_snapshot_status status = xe_session_snapshot_size(s, &size);
    if (status != XE_SNAPSHOT_OK) return status;
    uint64_t n = (uint64_t)xe_session_position(s);
    if (!n || n > 262144 || size <= n * KVSTORE_GLOBAL_ROW_SIZE * KVSTORE_GLOBAL_HEADS) return XE_SNAPSHOT_FORMAT;
    kvstore_shared_plan *p = calloc(1, sizeof *p);
    if (!p) return XE_SNAPSHOT_NOMEM;
    p->dir = dir; p->map.size = size; p->map.prefix = size - n * KVSTORE_GLOBAL_ROW_SIZE * KVSTORE_GLOBAL_HEADS;
    p->head_bytes = n * KVSTORE_GLOBAL_ROW_SIZE; p->map.group = *group;
    p->map.count = (size_t)((p->head_bytes + KVSTORE_CHUNK_SIZE - 1) / KVSTORE_CHUNK_SIZE) * KVSTORE_GLOBAL_HEADS;
    p->map.data_offset = KVSTORE_SHARED_HEADER_SIZE + p->map.count * KVSTORE_SHARED_REF_SIZE;
    p->reserve = p->map.data_offset + p->map.prefix;
    p->map.refs = calloc(p->map.count, sizeof *p->map.refs);
    p->missing = calloc(p->map.count, 1); p->created = calloc(p->map.count, 1);
    p->buffer = malloc(KVSTORE_CHUNK_SIZE);
    if (!p->map.refs || !p->missing || !p->created || !p->buffer) status = XE_SNAPSHOT_NOMEM;
    else status = kvstore_shared_stream(p, s);
    if (status != XE_SNAPSHOT_OK) { kvstore_shared_finish(p, 0); return status; }
    kvstore_shared_ref *unique = calloc(p->map.count, sizeof *unique);
    if (!unique) { kvstore_shared_finish(p, 0); return XE_SNAPSHOT_NOMEM; }
    size_t count = 0;
    for (size_t i = 0; i < p->map.count; i++) if (p->missing[i]) unique[count++] = p->map.refs[i];
    qsort(unique, count, sizeof *unique, kvstore_shared_ref_compare);
    for (size_t i = 0; i < count; i++)
        if (!i || memcmp(unique[i-1].hash, unique[i].hash, 32)) p->reserve += unique[i].length;
    free(unique);
    *out = p; return XE_SNAPSHOT_OK;
}
static xe_snapshot_status kvstore_shared_write(kvstore_shared_plan *p, xe_session *s, FILE *out) {
    p->writing = 1; p->output = fileno(out); p->position = 0; p->next = 0;
    uint8_t h[KVSTORE_SHARED_HEADER_SIZE] = {0}; memcpy(h, "XEKVSEG\n", 8);
    format_put_u32le(h + 8, 1); format_put_u32le(h + 12, KVSTORE_SHARED_HEADER_SIZE);
    format_put_u64le(h + 16, p->map.size); format_put_u64le(h + 24, p->map.prefix);
    memcpy(h + 32, p->map.group.bytes, 16); format_put_u64le(h + 48, p->map.count);
    uint64_t crc = 0;
    for (size_t i = 0; i < p->map.count; i++) {
        uint8_t r[KVSTORE_SHARED_REF_SIZE]; kvstore_shared_encode_ref(&p->map.refs[i], r); crc = format_crc64(crc, r, sizeof r);
        if (!kvstore_io_at(p->output, r, sizeof r, KVSTORE_SHARED_HEADER_SIZE + i * KVSTORE_SHARED_REF_SIZE, 1)) return XE_SNAPSHOT_IO;
    }
    format_put_u64le(h + 80, crc); format_put_u64le(h + 88, format_crc64(0, h, 88));
    if (!kvstore_io_at(p->output, h, sizeof h, 0, 1)) return XE_SNAPSHOT_IO;
    xe_snapshot_status status = kvstore_shared_stream(p, s);
    if (status == XE_SNAPSHOT_OK && (ftruncate(p->output, (off_t)(p->map.data_offset + p->map.prefix)) ||
                                    fsync(p->dir))) status = XE_SNAPSHOT_IO;
    return status;
}
static void kvstore_shared_finish(kvstore_shared_plan *p, int committed) {
    if (!p) return;
    if (!committed && p->writing) {
        for (size_t i = 0; i < p->map.count; i++) {
            char name[69], temporary[80];
            kvstore_shared_name(p->map.refs[i].hash, name);
            if (p->created[i]) unlinkat(p->dir, name, 0);
            snprintf(temporary, sizeof temporary, ".tmp-%s", name);
            if (p->missing[i]) unlinkat(p->dir, temporary, 0);
        }
        fsync(p->dir);
    }
    free(p->map.refs); free(p->missing); free(p->created); free(p->buffer); free(p);
}

typedef struct { FILE *source; int dir, chunk; size_t index; uint64_t position; kvstore_shared_map map; uint64_t *starts; } kvstore_shared_input;
static ssize_t kvstore_shared_read(void *opaque, char *data, size_t length) {
    kvstore_shared_input *r = opaque; size_t done = 0;
    if (r->position >= r->map.size) return 0;
    if (length > r->map.size - r->position) length = (size_t)(r->map.size - r->position);
    while (done < length) {
        uint64_t end, offset; int fd;
        if (r->position < r->map.prefix) {
            end = r->map.prefix; offset = r->map.data_offset + r->position; fd = fileno(r->source);
        } else {
            size_t lo = 0, hi = r->map.count;
            while (lo + 1 < hi) { size_t mid = lo + (hi-lo)/2; if (r->starts[mid] <= r->position) lo = mid; else hi = mid; }
            if (r->chunk < 0 || r->index != lo) {
                if (r->chunk >= 0) close(r->chunk);
                char name[69]; kvstore_shared_name(r->map.refs[lo].hash, name);
                r->chunk = openat(r->dir, name, O_RDONLY | O_NOFOLLOW | O_CLOEXEC); r->index = lo;
                struct stat st;
                if (r->chunk < 0 || fstat(r->chunk, &st) || !S_ISREG(st.st_mode) ||
                    st.st_size != (off_t)r->map.refs[lo].length) { errno = EIO; return -1; }
            }
            fd = r->chunk; offset = r->position - r->starts[lo]; end = r->starts[lo] + r->map.refs[lo].length;
        }
        size_t take = length - done; if (take > end - r->position) take = (size_t)(end - r->position);
        if (!kvstore_io_at(fd, data + done, take, offset, 0)) { errno = EIO; return -1; }
        done += take; r->position += take;
    }
    return (ssize_t)done;
}
static int kvstore_shared_seek(void *opaque, off64_t *off, int whence) {
    kvstore_shared_input *r = opaque;
    uint64_t base = whence == SEEK_SET ? 0 : whence == SEEK_CUR ? r->position : r->map.size;
    if (whence != SEEK_SET && whence != SEEK_CUR && whence != SEEK_END) return -1;
    if ((*off < 0 && (uint64_t)(-(*off + 1)) + 1 > base) ||
        (*off >= 0 && (uint64_t)*off > INT64_MAX - base)) return -1;
    r->position = *off < 0 ? base - ((uint64_t)(-(*off + 1)) + 1) : base + (uint64_t)*off;
    *off = (off64_t)r->position; return 0;
}
static int kvstore_shared_close(void *opaque) {
    kvstore_shared_input *r = opaque; int ok = fclose(r->source) == 0;
    if (r->chunk >= 0 && close(r->chunk)) ok = 0;
    free(r->map.refs); free(r->starts); free(r); return ok ? 0 : -1;
}
/* On success owns source. Standalone v1/v2 snapshots pass through unchanged. */
static xe_snapshot_status kvstore_shared_reader(int dir, FILE *source, FILE **out) {
    *out = NULL; kvstore_shared_input *r = calloc(1, sizeof *r);
    if (!r) return XE_SNAPSHOT_NOMEM;
    int kind = kvstore_shared_inspect(fileno(source), &r->map);
    if (kind <= 0) {
        free(r); if (kind < 0) return XE_SNAPSHOT_FORMAT;
        if (!format_seek(source, 0)) return XE_SNAPSHOT_IO;
        *out = source; return XE_SNAPSHOT_OK;
    }
    r->source = source; r->dir = dir; r->chunk = -1;
    r->starts = calloc(r->map.count, sizeof *r->starts);
    if (!r->starts) { free(r->map.refs); free(r); return XE_SNAPSHOT_NOMEM; }
    uint64_t start = r->map.prefix;
    for (size_t i = 0; i < r->map.count; i++) { r->starts[i] = start; start += r->map.refs[i].length; }
    cookie_io_functions_t io = {.read = kvstore_shared_read, .seek = kvstore_shared_seek, .close = kvstore_shared_close};
    *out = fopencookie(r, "rb", io);
    if (!*out) { free(r->map.refs); free(r->starts); free(r); return XE_SNAPSHOT_IO; }
    return XE_SNAPSHOT_OK;
}

static int kvstore_all_zero(const uint8_t *p, size_t length) {
    for (size_t i = 0; i < length; i++) if (p[i]) return 0;
    return 1;
}

static int64_t kvstore_now_ns(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) return 0;
    return (int64_t)now.tv_sec * INT64_C(1000000000) + now.tv_nsec;
}

static int kvstore_random(void *data, size_t length) {
    uint8_t *p = data;
    while (length) {
        ssize_t n = getrandom(p, length, 0);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        p += n;
        length -= (size_t)n;
    }
    return 1;
}

void kvstore_id_format(const kvstore_id *id, char out[33]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        out[2 * i] = hex[id->bytes[i] >> 4];
        out[2 * i + 1] = hex[id->bytes[i] & 15];
    }
    out[32] = '\0';
}

static int kvstore_hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

int kvstore_id_parse(const char *text, kvstore_id *id) {
    if (!text || !id || strlen(text) != 32) return 0;
    for (int i = 0; i < 16; i++) {
        int high = kvstore_hex_value(text[2 * i]);
        int low = kvstore_hex_value(text[2 * i + 1]);
        if (high < 0 || low < 0) return 0;
        id->bytes[i] = (uint8_t)(high << 4 | low);
    }
    return 1;
}

static void kvstore_names(const kvstore_id *id, char snapshot[37],
                          char meta[38]) {
    kvstore_id_format(id, snapshot);
    memcpy(snapshot + 32, ".xkv", 5);
    memcpy(meta, snapshot, 32);
    memcpy(meta + 32, ".meta", 6);
}

static int kvstore_snapshot_name(const char *name, kvstore_id *id) {
    if (strlen(name) != 36 || memcmp(name + 32, ".xkv", 5) != 0) return 0;
    char hex[33];
    memcpy(hex, name, 32);
    hex[32] = '\0';
    return kvstore_id_parse(hex, id);
}

static int kvstore_anchor_name(const char *name, char out[64]) {
    if (!name || !*name) return 0;
    size_t length = strlen(name);
    if (length > 48) return 0;
    for (size_t i = 0; i < length; i++) {
        char c = name[i];
        if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
              c == '-' || c == '_')) return 0;
    }
    memcpy(out, "anchor-", 7);
    memcpy(out + 7, name, length + 1);
    return 1;
}

static int kvstore_mkdirs(const char *path) {
    if (!path || !*path) return 0;
    char *copy = strdup(path);
    if (!copy) return 0;
    size_t length = strlen(copy);
    while (length > 1 && copy[length - 1] == '/') copy[--length] = '\0';
    for (char *p = copy + (copy[0] == '/'); *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (*copy && mkdir(copy, 0700) != 0 && errno != EEXIST) {
            free(copy);
            return 0;
        }
        *p = '/';
    }
    int ok = mkdir(copy, 0700) == 0 || errno == EEXIST;
    free(copy);
    return ok;
}

static int kvstore_write_all(int fd, const void *data, size_t length) {
    const uint8_t *p = data;
    while (length) {
        ssize_t n = write(fd, p, length);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        p += n;
        length -= (size_t)n;
    }
    return 1;
}

static int kvstore_read_all(int fd, void *data, size_t length) {
    uint8_t *p = data;
    while (length) {
        ssize_t n = read(fd, p, length);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        p += n;
        length -= (size_t)n;
    }
    return 1;
}

static int kvstore_temp_name(char out[64]) {
    kvstore_id id;
    if (!kvstore_random(id.bytes, sizeof id.bytes)) return 0;
    memcpy(out, ".tmp-", 5);
    kvstore_id_format(&id, out + 5);
    return 1;
}

static int kvstore_atomic_bytes(kvstore *store, const char *final_name,
                                const void *data, size_t length,
                                int durable) {
    char temporary[64];
    if (!kvstore_temp_name(temporary)) return 0;
    int fd = openat(store->directory_fd, temporary,
                    O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) return 0;
    int ok = kvstore_write_all(fd, data, length);
    if (ok && durable) ok = fsync(fd) == 0;
    if (close(fd) != 0) ok = 0;
    if (ok) ok = renameat(store->directory_fd, temporary,
                          store->directory_fd, final_name) == 0;
    if (ok && durable) ok = fsync(store->directory_fd) == 0;
    if (!ok) unlinkat(store->directory_fd, temporary, 0);
    return ok;
}

static void kvstore_meta_encode(const kvstore_meta *meta,
                                uint8_t out[KVSTORE_META_SIZE]) {
    memset(out, 0, KVSTORE_META_SIZE);
    memcpy(out, "XEKVMETA", 8);
    format_put_u32le(out + 8, 2);
    format_put_u32le(out + 12, meta->flags);
    memcpy(out + 16, meta->id.bytes, 16);
    memcpy(out + 32, meta->key, 32);
    format_put_u64le(out + 64, meta->payload_size);
    format_put_u64le(out + 72, meta->hits);
    format_put_u64le(out + 80, (uint64_t)meta->last_hit);
    format_put_u64le(out + 88, meta->rebuild_cost);
    format_put_u64le(out + 96, (uint64_t)meta->created);
    if (meta->flags & KVSTORE_META_GROUP) memcpy(out + 104, meta->group.bytes, 16);
    format_put_u64le(out + 120, format_crc64(0, out, 120));
}

static int kvstore_meta_decode(const uint8_t in[KVSTORE_META_SIZE],
                               const kvstore_id *id, kvstore_meta *meta) {
    if (memcmp(in, "XEKVMETA", 8) != 0 || (format_get_u32le(in + 8) != 1 && format_get_u32le(in + 8) != 2) ||
        format_get_u32le(in + 12) & ~(KVSTORE_META_KEY | KVSTORE_META_GROUP) ||
        memcmp(in + 16, id->bytes, 16) != 0 ||
        (format_get_u32le(in + 8) == 1 && (!kvstore_all_zero(in + 104, 16) ||
         (format_get_u32le(in + 12) & KVSTORE_META_GROUP))) ||
        format_get_u64le(in + 120) != format_crc64(0, in, 120)) return 0;
    memset(meta, 0, sizeof *meta);
    meta->id = *id;
    meta->flags = format_get_u32le(in + 12);
    memcpy(meta->key, in + 32, 32);
    meta->payload_size = format_get_u64le(in + 64);
    meta->hits = format_get_u64le(in + 72);
    meta->last_hit = (int64_t)format_get_u64le(in + 80);
    meta->rebuild_cost = format_get_u64le(in + 88);
    meta->created = (int64_t)format_get_u64le(in + 96);
    if (meta->flags & KVSTORE_META_GROUP) memcpy(meta->group.bytes, in + 104, 16);
    return 1;
}

static int kvstore_meta_read(kvstore *store, const kvstore_id *id,
                             kvstore_meta *meta) {
    char snapshot[37], name[38];
    kvstore_names(id, snapshot, name);
    int fd = openat(store->directory_fd, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return 0;
    struct stat st;
    uint8_t bytes[KVSTORE_META_SIZE];
    int ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
             st.st_size == KVSTORE_META_SIZE &&
             kvstore_read_all(fd, bytes, sizeof bytes);
    if (close(fd) != 0) ok = 0;
    return ok && kvstore_meta_decode(bytes, id, meta);
}

static int kvstore_meta_write(kvstore *store, const kvstore_meta *meta,
                              int durable) {
    char snapshot[37], name[38];
    kvstore_names(&meta->id, snapshot, name);
    uint8_t bytes[KVSTORE_META_SIZE];
    kvstore_meta_encode(meta, bytes);
    return kvstore_atomic_bytes(store, name, bytes, sizeof bytes, durable);
}

static void kvstore_anchor_encode(const kvstore_id *id,
                                  uint8_t out[KVSTORE_ANCHOR_SIZE]) {
    memset(out, 0, KVSTORE_ANCHOR_SIZE);
    memcpy(out, "XEKVANCH", 8);
    format_put_u32le(out + 8, 1);
    memcpy(out + 16, id->bytes, 16);
    format_put_u64le(out + 56, format_crc64(0, out, 56));
}

static int kvstore_anchor_decode(const uint8_t in[KVSTORE_ANCHOR_SIZE],
                                 kvstore_id *id) {
    if (memcmp(in, "XEKVANCH", 8) != 0 || format_get_u32le(in + 8) != 1 ||
        format_get_u32le(in + 12) != 0 || !kvstore_all_zero(in + 32, 24) ||
        format_get_u64le(in + 56) != format_crc64(0, in, 56)) return 0;
    memcpy(id->bytes, in + 16, 16);
    return 1;
}

static int kvstore_anchor_read_name(kvstore *store, const char *filename,
                                    kvstore_id *id) {
    int fd = openat(store->directory_fd, filename,
                    O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return errno == ENOENT ? 0 : -1;
    struct stat st;
    uint8_t bytes[KVSTORE_ANCHOR_SIZE];
    int ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
             st.st_size == KVSTORE_ANCHOR_SIZE &&
             kvstore_read_all(fd, bytes, sizeof bytes);
    if (close(fd) != 0) ok = 0;
    return ok && kvstore_anchor_decode(bytes, id) ? 1 : -1;
}

static int kvstore_id_equal(const kvstore_id *a, const kvstore_id *b) {
    return memcmp(a->bytes, b->bytes, sizeof a->bytes) == 0;
}

static int kvstore_collect_anchors(kvstore *store, kvstore_id **ids,
                                   size_t *count) {
    *ids = NULL;
    *count = 0;
    int duplicate = openat(store->directory_fd, ".",
                           O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (duplicate < 0) return 0;
    DIR *directory = fdopendir(duplicate);
    if (!directory) {
        close(duplicate);
        return 0;
    }
    size_t capacity = 0;
    struct dirent *entry;
    int ok = 1;
    for (;;) {
        errno = 0;
        entry = readdir(directory);
        if (!entry) { if (errno) ok = 0; break; }
        if (strncmp(entry->d_name, "anchor-", 7) != 0) continue;
        kvstore_id id;
        if (kvstore_anchor_read_name(store, entry->d_name, &id) != 1) continue;
        if (*count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            kvstore_id *grown = realloc(*ids, next * sizeof(**ids));
            if (!grown) {
                ok = 0;
                break;
            }
            *ids = grown;
            capacity = next;
        }
        (*ids)[(*count)++] = id;
    }
    closedir(directory);
    if (!ok) {
        free(*ids);
        *ids = NULL;
        *count = 0;
    }
    return ok;
}

static int kvstore_id_anchored(const kvstore_id *id, const kvstore_id *anchors,
                               size_t count) {
    for (size_t i = 0; i < count; i++)
        if (kvstore_id_equal(id, &anchors[i])) return 1;
    return 0;
}

static void kvstore_candidates_free(kvstore_candidate *c, size_t count) {
    for (size_t i = 0; i < count; i++) free(c[i].map.refs);
    free(c);
}

kvstore_status kvstore_set_group(kvstore *s, const kvstore_id *id, const kvstore_id *group) {
    if (!s || !id || !group) return KVSTORE_INVALID_ARGUMENT;
    if (!s->budget) return KVSTORE_DISABLED;
    for (size_t i = 0; i < s->group_count; i++) if (kvstore_id_equal(id, &s->groups[i].id)) {
        s->groups[i].group = *group; return KVSTORE_OK;
    }
    void *grown = realloc(s->groups, (s->group_count + 1) * sizeof *s->groups);
    if (!grown) return KVSTORE_IO;
    s->groups = grown;
    s->groups[s->group_count].id = *id; s->groups[s->group_count++].group = *group;
    return KVSTORE_OK;
}

typedef struct { uint8_t hash[32]; size_t owner; } chunk_use;
static int chunk_compare(const void *a, const void *b) {
    const chunk_use *x = a, *y = b;
    int cmp = memcmp(x->hash, y->hash, 32);
    return cmp ? cmp : (x->owner > y->owner) - (x->owner < y->owner);
}
static size_t chunk_lower(const chunk_use *uses, size_t count, const uint8_t hash[32]) {
    size_t lo = 0, hi = count;
    while (lo < hi) { size_t mid = lo + (hi-lo)/2;
        if (memcmp(uses[mid].hash, hash, 32) < 0) lo = mid + 1; else hi = mid;
    }
    return lo;
}
/* Count physical files once. The pending plan holds reused chunks alive
 * during reservation and garbage collection. */
static int kvstore_account(kvstore *s, kvstore_candidate *c, size_t count,
                            const kvstore_shared_map *plan, int cleanup,
                            uint64_t *total, uint64_t *pinned) {
    size_t n = plan ? plan->count : 0; int valid = 1;
    for (size_t i = 0; i < count; i++) { n += c[i].map.count; if (c[i].invalid) valid = 0; }
    chunk_use *uses = calloc(n ? n : 1, sizeof *uses);
    if (!uses) return 0;
    size_t at = 0;
    for (size_t i = 0; i <= count; i++) {
        const kvstore_shared_map *map = i < count ? &c[i].map : plan;
        if (!map) continue;
        for (size_t j = 0; j < map->count; j++) {
            memcpy(uses[at].hash, map->refs[j].hash, 32); uses[at++].owner = i;
        }
    }
    qsort(uses, n, sizeof *uses, chunk_compare);
    if (cleanup && fsync(s->directory_fd)) { free(uses); return 0; }
    int fd = openat(s->directory_fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    DIR *dir = fd < 0 ? NULL : fdopendir(fd);
    if (!dir) { if (fd >= 0) close(fd); free(uses); return 0; }
    *total = *pinned = 0;
    for (size_t i = 0; i < count; i++) {
        if (c[i].pinned) *pinned += c[i].bytes;
    }
    int ok = 1, changed = 0; struct dirent *entry;
    for (;;) {
        errno = 0;
        entry = readdir(dir);
        if (!entry) { if (errno) ok = 0; break; }
        const char *name = entry->d_name; uint8_t hash[32]; kvstore_id id;
        int chunk = kvstore_shared_parse_name(name, hash);
        size_t len = strlen(name);
        int metadata = len == 37 && !strcmp(name + 32, ".meta");
        if (!chunk && !metadata && !kvstore_snapshot_name(name, &id) &&
            strncmp(name, "anchor-", 7) && strncmp(name, ".tmp-", 5)) continue;
        struct stat st;
        if (fstatat(s->directory_fd, name, &st, AT_SYMLINK_NOFOLLOW) ||
            !S_ISREG(st.st_mode) || st.st_size < 0) { ok = 0; break; }
        if (chunk) {
            size_t first = chunk_lower(uses, n, hash), end = first;
            while (end < n && !memcmp(uses[end].hash, hash, 32)) end++;
            if (first == end && cleanup && valid) {
                if (unlinkat(s->directory_fd, name, 0)) { ok = 0; break; }
                changed = 1; continue;
            }
            if (first != end) {
                int held = 0;
                for (size_t j = first; j < end; j++)
                    if (uses[j].owner == count || c[uses[j].owner].pinned) { held = 1; break; }
                if (held) *pinned += (uint64_t)st.st_size;
            }
        }
        if (metadata && cleanup) {
            char snapshot[37]; memcpy(snapshot, name, 32); memcpy(snapshot+32, ".xkv", 5);
            struct stat target;
            if (fstatat(s->directory_fd, snapshot, &target, AT_SYMLINK_NOFOLLOW) && errno == ENOENT) {
                if (unlinkat(s->directory_fd, name, 0)) { ok = 0; break; }
                changed = 1; continue;
            }
        }
        if (*total > UINT64_MAX - (uint64_t)st.st_size) { ok = 0; break; }
        *total += (uint64_t)st.st_size;
    }
    closedir(dir); free(uses);
    if (changed && fsync(s->directory_fd)) ok = 0;
    return ok;
}

static int kvstore_collect_candidates(kvstore *store, kvstore_candidate **out,
                                      size_t *count, uint64_t *total, uint64_t *pinned,
                                      const kvstore_shared_map *plan, int cleanup) {
    *out = NULL; *count = 0;
    kvstore_id *anchors = NULL; size_t anchor_count = 0;
    if (!kvstore_collect_anchors(store, &anchors, &anchor_count)) return 0;
    int fd = openat(store->directory_fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    DIR *dir = fd < 0 ? NULL : fdopendir(fd);
    if (!dir) { if (fd >= 0) close(fd); free(anchors); return 0; }
    size_t capacity = 0; int ok = 1; struct dirent *entry;
    for (;;) {
        errno = 0;
        entry = readdir(dir);
        if (!entry) { if (errno) ok = 0; break; }
        kvstore_id id;
        if (!kvstore_snapshot_name(entry->d_name, &id)) continue;
        if (*count == capacity) {
            size_t next = capacity ? capacity * 2 : 16;
            void *grown = realloc(*out, next * sizeof **out);
            if (!grown) { ok = 0; break; }
            *out = grown; capacity = next;
        }
        kvstore_candidate *c = &(*out)[(*count)++]; memset(c, 0, sizeof *c);
        c->meta.id = id;
        kvstore_meta_read(store, &id, &c->meta);
        kvstore_names(&id, c->snapshot_name, c->meta_name);
        int snapshot = openat(store->directory_fd, c->snapshot_name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        struct stat st;
        if (snapshot < 0 || fstat(snapshot, &st) || !S_ISREG(st.st_mode) || st.st_size < 0) {
            if (snapshot >= 0) close(snapshot);
            ok = 0; break;
        }
        c->meta.payload_size = c->bytes = (uint64_t)st.st_size;
        c->used = (int64_t)st.st_mtim.tv_sec * INT64_C(1000000000) + st.st_mtim.tv_nsec;
        /* Import recency from old sidecars; new accesses touch the manifest. */
        if (c->meta.last_hit > c->used) c->used = c->meta.last_hit;
        if (c->meta.created > c->used) c->used = c->meta.created;
        int kind = kvstore_shared_inspect(snapshot, &c->map); close(snapshot);
        c->invalid = kind < 0;
        if (kind == 1) { c->meta.group = c->map.group; c->meta.flags |= KVSTORE_META_GROUP; }
        for (size_t i = 0; i < store->group_count; i++)
            if (kvstore_id_equal(&id, &store->groups[i].id)) {
                c->meta.group = store->groups[i].group; c->meta.flags |= KVSTORE_META_GROUP; break;
            }
        if (!fstatat(store->directory_fd, c->meta_name, &st, AT_SYMLINK_NOFOLLOW) &&
            S_ISREG(st.st_mode) && st.st_size >= 0) c->bytes += (uint64_t)st.st_size;
        c->pinned = kvstore_id_anchored(&id, anchors, anchor_count) ||
            (store->protect_id && kvstore_id_equal(&id, store->protect_id)) ||
            (store->protect_group && (c->meta.flags & KVSTORE_META_GROUP) &&
             kvstore_id_equal(&c->meta.group, store->protect_group));
    }
    closedir(dir); free(anchors);
    if (ok) ok = kvstore_account(store, *out, *count, plan, cleanup, total, pinned);
    if (!ok) { kvstore_candidates_free(*out, *count); *out = NULL; *count = 0; }
    return ok;
}

static int same_group(const kvstore_meta *a, const kvstore_meta *b) {
    return (a->flags & b->flags & KVSTORE_META_GROUP) && kvstore_id_equal(&a->group, &b->group);
}
/* Stable tie breaking, independent of readdir order. */
static int kvstore_newer(const kvstore_candidate *a, const kvstore_candidate *b) {
    return a->meta.created != b->meta.created ? a->meta.created > b->meta.created :
        memcmp(a->meta.id.bytes, b->meta.id.bytes, 16) > 0;
}

static int kvstore_group_order(const kvstore_meta *a, const kvstore_meta *b) {
    const kvstore_id *x = a->flags & KVSTORE_META_GROUP ? &a->group : &a->id;
    const kvstore_id *y = b->flags & KVSTORE_META_GROUP ? &b->group : &b->id;
    int order = memcmp(x->bytes, y->bytes, 16);
    if (order) return order;
    /* A standalone ID may equal a group ID; keep them distinct and stable. */
    return !!(a->flags & KVSTORE_META_GROUP) - !!(b->flags & KVSTORE_META_GROUP);
}

/* Keep the nearest available point at/below half, plus the first saved point
 * at/above twice that reserve. Before the first doubling, keep the earliest
 * available predecessor. Never discard a milestone just because a nearby
 * latest point replaced it. Missing milestones are not manufactured. */
static void kvstore_reserves(const kvstore_candidate *c, size_t count,
                              size_t newest, size_t *reserve, size_t *future) {
    uint64_t n = c[newest].meta.rebuild_cost;
    size_t earliest = count;
    *reserve = *future = count;
    for (size_t i = 0; i < count; i++) {
        uint64_t at = c[i].meta.rebuild_cost;
        if (i == newest || c[i].invalid || !same_group(&c[newest].meta, &c[i].meta) ||
            !at || at >= n) continue;
        if (earliest == count || at < c[earliest].meta.rebuild_cost ||
            (at == c[earliest].meta.rebuild_cost && kvstore_newer(&c[i], &c[earliest]))) earliest = i;
        if (at <= n / 2 && (*reserve == count || at > c[*reserve].meta.rebuild_cost ||
            (at == c[*reserve].meta.rebuild_cost && kvstore_newer(&c[i], &c[*reserve])))) *reserve = i;
    }
    if (*reserve == count) *reserve = earliest;
    if (*reserve == count) return;
    uint64_t base = c[*reserve].meta.rebuild_cost;
    for (size_t i = 0; i < count; i++) {
        uint64_t at = c[i].meta.rebuild_cost;
        if (i == newest || c[i].invalid || !same_group(&c[newest].meta, &c[i].meta) ||
            at >= n || at / 2 < base) continue;
        if (*future == count || at < c[*future].meta.rebuild_cost ||
            (at == c[*future].meta.rebuild_cost && kvstore_newer(&c[i], &c[*future]))) *future = i;
    }
}

/* Recency lives in the manifest inode, so an access needs no extra disk
 * reservation or sidecar rewrite. Carry it to a surviving reserve BEFORE
 * removing the latest point, including across crashes and reopen. */
static int kvstore_touch(kvstore *s, const char *name, int64_t used) {
    int fd = openat(s->directory_fd, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return 0;
    struct stat st;
    int ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode);
    if (ok) {
        int64_t previous = (int64_t)st.st_mtim.tv_sec * INT64_C(1000000000) + st.st_mtim.tv_nsec;
        if (used < previous) used = previous;
        struct timespec times[2] = {{.tv_nsec = UTIME_OMIT},
            {.tv_sec = used / INT64_C(1000000000), .tv_nsec = used % INT64_C(1000000000)}};
        ok = futimens(fd, times) == 0 && fsync(fd) == 0;
    }
    if (close(fd)) ok = 0;
    return ok;
}

static int kvstore_delete_candidate(kvstore *s, const kvstore_candidate *c) {
    if (unlinkat(s->directory_fd, c->snapshot_name, 0) && errno != ENOENT) return 0;
    /* Persist removal of the manifest BEFORE reclaiming any referenced chunk. */
    if (fsync(s->directory_fd)) return 0;
    if (unlinkat(s->directory_fd, c->meta_name, 0) && errno != ENOENT) return 0;
    return fsync(s->directory_fd) == 0 && !kvstore_fault("checkpoint-unlinked");
}
static kvstore_status kvstore_make_room(kvstore *s, uint64_t reserve,
                                        const kvstore_shared_map *plan) {
    if (!s->budget) return KVSTORE_DISABLED;
    for (;;) {
        kvstore_candidate *c; size_t count; uint64_t total, pinned;
        if (!kvstore_collect_candidates(s, &c, &count, &total, &pinned, plan, 1)) return KVSTORE_IO;
        kvstore_status status = KVSTORE_OK;
        size_t victim = count;
        if (reserve > s->budget || pinned > s->budget - reserve) status = KVSTORE_BUDGET;
        else if (total > s->budget || reserve > s->budget - total) {
            int64_t oldest = INT64_MAX, recent = 0;
            size_t newest = count, reserve = count, future = count;
            for (size_t i = 0; i < count; i++) if (!c[i].pinned) {
                /* Unreadable manifests cannot provide a useful reserve. */
                if (c[i].invalid) { victim = i; break; }
                int held = 0; int64_t used = c[i].used;
                for (size_t j = 0; j < count; j++) if (same_group(&c[i].meta, &c[j].meta)) {
                    if (c[j].pinned) held = 1;
                    if (c[j].used > used) used = c[j].used;
                }
                if (held) continue;
                if (victim == count || used < oldest || (used == oldest &&
                    kvstore_group_order(&c[i].meta, &c[victim].meta) < 0)) {
                    victim = i; oldest = used;
                }
            }
            if (victim == count) status = KVSTORE_BUDGET;
            else if (c[victim].invalid) {
                if (!kvstore_delete_candidate(s, &c[victim])) status = KVSTORE_IO;
            } else {
                newest = victim; recent = c[victim].used;
                for (size_t i = 0; i < count; i++) if (same_group(&c[victim].meta, &c[i].meta)) {
                    if (kvstore_newer(&c[i], &c[newest])) newest = i;
                    if (c[i].used > recent) recent = c[i].used;
                }
                kvstore_reserves(c, count, newest, &reserve, &future);
                if (reserve != count && (!kvstore_touch(s, c[reserve].snapshot_name, recent) ||
                    kvstore_fault("reserve-carried"))) status = KVSTORE_IO;
                /* A reduction is one stage: remove the future and latest,
                 * retaining only the reserve even if the first unlink suffices.
                 * Delete latest last so a partial failure retains a good head. */
                for (size_t i = 0; status == KVSTORE_OK && i < count; i++)
                    if (i != newest && i != reserve && same_group(&c[newest].meta, &c[i].meta) &&
                        !kvstore_delete_candidate(s, &c[i])) status = KVSTORE_IO;
                if (status == KVSTORE_OK && !kvstore_delete_candidate(s, &c[newest])) status = KVSTORE_IO;
            }
        }
        kvstore_candidates_free(c, count);
        if (status != KVSTORE_OK || victim == count) return status;
    }
}

static void kvstore_cleanup_temps(kvstore *store) {
    int duplicate = openat(store->directory_fd, ".",
                           O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (duplicate < 0) return;
    DIR *directory = fdopendir(duplicate);
    if (!directory) {
        close(duplicate);
        return;
    }
    int changed = 0;
    struct dirent *entry;
    while ((entry = readdir(directory)) != NULL) {
        if (strncmp(entry->d_name, ".tmp-", 5) != 0) continue;
        if (unlinkat(store->directory_fd, entry->d_name, 0) == 0) changed = 1;
    }
    closedir(directory);
    if (changed) fsync(store->directory_fd);
}

kvstore_status kvstore_open(kvstore **out, const char *directory,
                            uint64_t budget) {
    if (!out || !directory || !*directory) return KVSTORE_INVALID_ARGUMENT;
    *out = NULL;
    if (!budget) {
        kvstore *disabled = calloc(1, sizeof *disabled);
        if (!disabled) return KVSTORE_IO;
        disabled->directory_fd = disabled->lock_fd = -1;
        *out = disabled; return KVSTORE_OK;
    }
    if (!kvstore_mkdirs(directory)) return KVSTORE_IO;
    int fd = open(directory, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return KVSTORE_IO;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISDIR(st.st_mode) ||
        fchmod(fd, 0700) != 0) {
        close(fd);
        return KVSTORE_IO;
    }
    kvstore *store = calloc(1, sizeof *store);
    if (!store) {
        close(fd);
        return KVSTORE_IO;
    }
    store->directory_fd = fd;
    store->budget = budget;
    store->lock_fd = openat(fd, ".lock", O_RDWR | O_CREAT | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (store->lock_fd < 0 || flock(store->lock_fd, LOCK_EX | LOCK_NB)) {
        kvstore_close(store); return KVSTORE_IO;
    }
    kvstore_cleanup_temps(store);
    *out = store;
    return KVSTORE_OK;
}

void kvstore_close(kvstore *store) {
    if (!store) return;
    if (store->directory_fd >= 0) close(store->directory_fd);
    if (store->lock_fd >= 0) close(store->lock_fd);
    free(store->groups);
    free(store);
}

kvstore_status kvstore_save(kvstore *store, xe_session *session,
                            const kvstore_save_options *options,
                            kvstore_id *id, xe_snapshot_status *snapshot_status) {
    if (snapshot_status) *snapshot_status = XE_SNAPSHOT_OK;
    if (!store || !session || !id) return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) return KVSTORE_DISABLED;
    kvstore_save_options defaults = {0};
    if (!options) options = &defaults;
    if ((options->has_key != 0 && options->has_key != 1) ||
        (options->has_group != 0 && options->has_group != 1) ||
        (options->pinned != 0 && options->pinned != 1)) return KVSTORE_INVALID_ARGUMENT;
    uint64_t size = 0; kvstore_shared_plan *plan = NULL;
    xe_snapshot_status core = options->has_group
        ? kvstore_shared_prepare(store->directory_fd, session, &options->group, &plan)
        : xe_session_snapshot_size(session, &size);
    if (core != XE_SNAPSHOT_OK) {
        if (snapshot_status) *snapshot_status = core;
        return KVSTORE_REJECTED;
    }
    if (plan) size = plan->reserve;
    if (size > UINT64_MAX - 2 * KVSTORE_META_SIZE) { kvstore_shared_finish(plan, 0); return KVSTORE_BUDGET; }
    store->protect_group = options->has_group ? &options->group : NULL;
    kvstore_status status = kvstore_make_room(store, size + 2 * KVSTORE_META_SIZE, plan ? &plan->map : NULL);
    store->protect_group = NULL;
    if (status != KVSTORE_OK) { kvstore_shared_finish(plan, 0); return status; }
    char snapshot[37], meta_name[38], temporary[64] = {0};
    int fd = -1, published = 0;
    for (int attempt = 0; attempt < 32; attempt++) {
        if (!kvstore_random(id->bytes, sizeof id->bytes)) goto failed;
        kvstore_names(id, snapshot, meta_name);
        if (faccessat(store->directory_fd, snapshot, F_OK, 0) == 0) continue;
        if (errno != ENOENT || !kvstore_temp_name(temporary)) goto failed;
        fd = openat(store->directory_fd, temporary, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (fd >= 0) break;
        if (errno != EEXIST) goto failed;
    }
    if (fd < 0) goto failed;
    FILE *file = fdopen(fd, "w+b");
    if (!file) { close(fd); goto failed; }
    core = plan ? kvstore_shared_write(plan, session, file) : xe_session_snapshot_save(session, file);
    int ok = core == XE_SNAPSHOT_OK && fflush(file) == 0;
    if (ok && kvstore_fault("snapshot-written")) ok = 0;
    if (ok) ok = fsync(fileno(file)) == 0;
    if (ok && kvstore_fault("snapshot-synced")) ok = 0;
    if (fclose(file)) ok = 0;
    if (!ok) goto failed;
    if (renameat2(store->directory_fd, temporary, store->directory_fd, snapshot, RENAME_NOREPLACE)) goto failed;
    published = 1;
    if (kvstore_fault("snapshot-renamed") || fsync(store->directory_fd) ||
        kvstore_fault("snapshot-directory-synced")) goto failed;
    kvstore_meta meta = {0};
    meta.id = *id;
    meta.payload_size = plan ? plan->map.data_offset + plan->map.prefix : size;
    meta.rebuild_cost = options->rebuild_cost; meta.created = kvstore_now_ns();
    if (options->has_key) { meta.flags |= KVSTORE_META_KEY; memcpy(meta.key, options->key, 32); }
    if (options->has_group) { meta.flags |= KVSTORE_META_GROUP; meta.group = options->group; }
    if (!kvstore_meta_write(store, &meta, 1)) goto failed;
    kvstore_shared_finish(plan, 1); return KVSTORE_OK;
failed:
#ifdef XE_TEST_CACHE_TRACE
    fprintf(stderr, "cache save: published=%d core=%s errno=%d (%s)\n",
            published, xe_snapshot_status_name(core), errno, strerror(errno));
#endif
    if (*temporary) unlinkat(store->directory_fd, temporary, 0);
    if (published) {
        /* If durable removal fails, keep chunks: the manifest might survive. */
        if (unlinkat(store->directory_fd, snapshot, 0) == 0 && fsync(store->directory_fd) == 0) {
            published = 0; unlinkat(store->directory_fd, meta_name, 0);
        }
    }
    kvstore_shared_finish(plan, published);
    if (snapshot_status) *snapshot_status = core;
    return core == XE_SNAPSHOT_OK || core == XE_SNAPSHOT_IO ? KVSTORE_IO : KVSTORE_REJECTED;
}

/* Validate the reserve against the live prefix without loading KV or changing
 * the session. A rewound/diverged branch cannot become a useful reserve. */
static int kvstore_prefix(kvstore *store, const kvstore_candidate *c, xe_session *session) {
    int fd = openat(store->directory_fd, c->snapshot_name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return 0;
    FILE *source = fdopen(fd, "rb"), *reader = NULL;
    if (!source) { close(fd); return 0; }
    xe_snapshot_status status = kvstore_shared_reader(store->directory_fd, source, &reader);
    if (status == XE_SNAPSHOT_OK) {
        source = reader;
        status = xe_session_snapshot_check_prefix(session, reader, c->meta.rebuild_cost);
    }
    if (fclose(source)) return 0;
    return status == XE_SNAPSHOT_OK;
}

kvstore_status kvstore_committed(kvstore *store, const kvstore_id *id, xe_session *session,
                                  kvstore_eligible_fn eligible, void *context) {
    if (!store || !id || !session) return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) return KVSTORE_DISABLED;
    kvstore_candidate *c; size_t count; uint64_t total, pinned;
    if (!kvstore_collect_candidates(store, &c, &count, &total, &pinned, NULL, 0)) return KVSTORE_IO;
    size_t newest = count;
    for (size_t i = 0; i < count; i++) if (kvstore_id_equal(id, &c[i].meta.id)) newest = i;
    if (newest == count) { kvstore_candidates_free(c, count); return KVSTORE_MISS; }
    if (!kvstore_prefix(store, &c[newest], session)) {
        kvstore_candidates_free(c, count); return KVSTORE_REJECTED;
    }
    int64_t recent = c[newest].used;
    for (size_t i = 0; i < count; i++) if (same_group(&c[newest].meta, &c[i].meta)) {
        if (c[i].used > recent) recent = c[i].used;
        if (i != newest && ((eligible && !eligible(&c[i].meta.id, context)) ||
            !kvstore_prefix(store, &c[i], session))) c[i].invalid = 1;
    }
    size_t reserve, future;
    kvstore_reserves(c, count, newest, &reserve, &future);
    kvstore_status status = kvstore_touch(store, c[newest].snapshot_name, recent) ? KVSTORE_OK : KVSTORE_IO;
    for (size_t i = 0; status == KVSTORE_OK && i < count; i++) {
        if (c[i].pinned || !same_group(&c[newest].meta, &c[i].meta)) continue;
        if (i != newest && i != reserve && i != future && !kvstore_delete_candidate(store, &c[i])) status = KVSTORE_IO;
    }
    kvstore_candidates_free(c, count);
    store->protect_id = id;
    if (status == KVSTORE_OK) status = kvstore_make_room(store, 0, NULL);
    store->protect_id = NULL;
    return status;
}

static kvstore_status kvstore_load_impl(kvstore *store, const kvstore_id *id,
                            xe_session *session, const xe_tokens *expected, const xe_input *mixed,
                            xe_snapshot_status *snapshot_status) {
    if (snapshot_status) *snapshot_status = XE_SNAPSHOT_OK;
    if (!store || !id || !session || !expected)
        return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) return KVSTORE_DISABLED;
    char snapshot_name[37], meta_name[38];
    kvstore_names(id, snapshot_name, meta_name);
    int fd = openat(store->directory_fd, snapshot_name,
                    O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (fd < 0) return errno == ENOENT ? KVSTORE_MISS : KVSTORE_IO;
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode)) {
        close(fd);
        return KVSTORE_IO;
    }
    FILE *file = fdopen(fd, "rb");
    if (!file) {
        close(fd);
        return KVSTORE_IO;
    }
    FILE *reader = NULL;
    xe_snapshot_status core = kvstore_shared_reader(store->directory_fd, file, &reader);
    if (core == XE_SNAPSHOT_OK) {
        file = reader;
        core = mixed ? xe_session_snapshot_load_input(session,file,mixed) :
                       xe_session_snapshot_load(session,file,expected);
    }
    int close_ok = fclose(file) == 0;
    if (snapshot_status) *snapshot_status = core;
    if (!close_ok) return KVSTORE_IO;
    if (core != XE_SNAPSHOT_OK) return KVSTORE_REJECTED;

    if (!kvstore_touch(store, snapshot_name, kvstore_now_ns())) return KVSTORE_IO;
    return KVSTORE_OK;
}

kvstore_status kvstore_load(kvstore *store,const kvstore_id *id,xe_session *session,
                            const xe_tokens *expected,xe_snapshot_status *status) {
    return kvstore_load_impl(store,id,session,expected,NULL,status);
}

kvstore_status kvstore_load_input(kvstore *store,const kvstore_id *id,xe_session *session,
                                  const xe_input *input,xe_snapshot_status *status) {
    if(!input){if(status)*status=XE_SNAPSHOT_INVALID_ARGUMENT;return KVSTORE_INVALID_ARGUMENT;}
    xe_tokens expected={(int32_t *)input->tokens,input->count,input->count};
    return kvstore_load_impl(store,id,session,&expected,input,status);
}

kvstore_status kvstore_find(kvstore *store, const uint8_t key[32],
                            kvstore_id *id) {
    if (!store || !key || !id) return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) return KVSTORE_DISABLED;
    kvstore_candidate *candidates;
    size_t count;
    uint64_t total, pinned;
    if (!kvstore_collect_candidates(store, &candidates, &count, &total,
                                    &pinned, NULL, 0)) return KVSTORE_IO;
    int found = 0;
    kvstore_meta best;
    for (size_t i = 0; i < count; i++) {
        kvstore_meta *meta = &candidates[i].meta;
        if (!(meta->flags & KVSTORE_META_KEY) ||
            memcmp(meta->key, key, 32) != 0) continue;
        if (!found || meta->created > best.created ||
            (meta->created == best.created &&
             memcmp(meta->id.bytes, best.id.bytes, 16) < 0)) {
            best = *meta;
            found = 1;
        }
    }
    if (found) *id = best.id;
    kvstore_candidates_free(candidates, count);
    return found ? KVSTORE_OK : KVSTORE_MISS;
}

kvstore_status kvstore_anchor_get(kvstore *store, const char *name,
                                  kvstore_id *id) {
    if (!store || !id) return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) return KVSTORE_DISABLED;
    char filename[64];
    if (!kvstore_anchor_name(name, filename)) return KVSTORE_INVALID_ARGUMENT;
    int result = kvstore_anchor_read_name(store, filename, id);
    return result == 1 ? KVSTORE_OK : result == 0 ? KVSTORE_MISS : KVSTORE_IO;
}

kvstore_status kvstore_anchor_set(kvstore *store, const char *name,
                                  const kvstore_id *id) {
    if (!store || !id) return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) return KVSTORE_DISABLED;
    char filename[64], snapshot[37], meta_name[38];
    if (!kvstore_anchor_name(name, filename)) return KVSTORE_INVALID_ARGUMENT;
    kvstore_names(id, snapshot, meta_name);
    struct stat st;
    if (fstatat(store->directory_fd, snapshot, &st, AT_SYMLINK_NOFOLLOW) != 0)
        return errno == ENOENT ? KVSTORE_MISS : KVSTORE_IO;
    if (!S_ISREG(st.st_mode)) return KVSTORE_IO;
    store->protect_id = id;
    kvstore_status room = kvstore_make_room(store, KVSTORE_ANCHOR_SIZE, NULL);
    store->protect_id = NULL;
    if (room != KVSTORE_OK) return room;
    uint8_t bytes[KVSTORE_ANCHOR_SIZE];
    kvstore_anchor_encode(id, bytes);
    if (!kvstore_atomic_bytes(store, filename, bytes, sizeof bytes, 1))
        return KVSTORE_IO;
    return kvstore_make_room(store, 0, NULL);
}

kvstore_status kvstore_anchor_clear(kvstore *store, const char *name) {
    if (!store) return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) return KVSTORE_DISABLED;
    char filename[64];
    if (!kvstore_anchor_name(name, filename)) return KVSTORE_INVALID_ARGUMENT;
    if (unlinkat(store->directory_fd, filename, 0) != 0)
        return errno == ENOENT ? KVSTORE_MISS : KVSTORE_IO;
    if (fsync(store->directory_fd) != 0) return KVSTORE_IO;
    return kvstore_make_room(store, 0, NULL);
}

kvstore_status kvstore_contains(kvstore *store, const kvstore_id *id) {
    if (!store || !id) return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) return KVSTORE_DISABLED;
    char snapshot[37], meta[38]; kvstore_names(id, snapshot, meta);
    struct stat st;
    if (fstatat(store->directory_fd, snapshot, &st, AT_SYMLINK_NOFOLLOW))
        return errno == ENOENT ? KVSTORE_MISS : KVSTORE_IO;
    return S_ISREG(st.st_mode) ? KVSTORE_OK : KVSTORE_IO;
}

kvstore_status kvstore_remove(kvstore *store, const kvstore_id *id) {
    if (!store || !id) return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) return KVSTORE_DISABLED;
    kvstore_id *anchors = NULL;
    size_t anchor_count = 0;
    if (!kvstore_collect_anchors(store, &anchors, &anchor_count))
        return KVSTORE_IO;
    int anchored = kvstore_id_anchored(id, anchors, anchor_count);
    free(anchors);
    if (anchored) return KVSTORE_PINNED;
    char snapshot[37], meta_name[38];
    kvstore_names(id, snapshot, meta_name);
    if (unlinkat(store->directory_fd, snapshot, 0) != 0)
        return errno == ENOENT ? KVSTORE_MISS : KVSTORE_IO;
    if (fsync(store->directory_fd)) return KVSTORE_IO;
    unlinkat(store->directory_fd, meta_name, 0);
    if (fsync(store->directory_fd)) return KVSTORE_IO;
    return kvstore_make_room(store, 0, NULL);
}

kvstore_status kvstore_enforce_budget(kvstore *store) {
    if (!store) return KVSTORE_INVALID_ARGUMENT;
    return kvstore_make_room(store, 0, NULL);
}

kvstore_status kvstore_usage(kvstore *store, uint64_t *total,
                             uint64_t *pinned) {
    if (!store || !total || !pinned) return KVSTORE_INVALID_ARGUMENT;
    if (!store->budget) { *total = *pinned = 0; return KVSTORE_DISABLED; }
    kvstore_candidate *candidates;
    size_t count;
    if (!kvstore_collect_candidates(store, &candidates, &count, total, pinned, NULL, 0))
        return KVSTORE_IO;
    kvstore_candidates_free(candidates, count);
    return KVSTORE_OK;
}

const char *kvstore_status_name(kvstore_status status) {
    static const char *names[] = {
        "ok", "miss", "disabled", "invalid-argument", "io", "budget",
        "rejected", "pinned"
    };
    return (unsigned)status < sizeof names / sizeof names[0]
           ? names[status] : "unknown";
}
