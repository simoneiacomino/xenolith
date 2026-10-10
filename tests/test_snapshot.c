#define XE_TEST_SESSION
#include "../xenolith.c"

#include <sys/stat.h>

size_t xe_test_prefill_batches;
size_t xe_test_decode_tokens;

typedef struct {
    int fd;
    off64_t position;
    uint64_t read_bytes;
    uint64_t fail_after;
} snapshot_fault_cookie;

static ssize_t snapshot_fault_read(void *opaque, char *buffer, size_t length) {
    snapshot_fault_cookie *cookie = opaque;
    if (cookie->read_bytes >= cookie->fail_after) {
        errno = EIO;
        return -1;
    }
    uint64_t available = cookie->fail_after - cookie->read_bytes;
    if (length > available) length = (size_t)available;
    ssize_t n = pread(cookie->fd, buffer, length, cookie->position);
    if (n > 0) {
        cookie->position += n;
        cookie->read_bytes += (uint64_t)n;
    }
    return n;
}

static int snapshot_fault_seek(void *opaque, off64_t *offset, int whence) {
    snapshot_fault_cookie *cookie = opaque;
    off64_t base;
    if (whence == SEEK_SET) base = 0;
    else if (whence == SEEK_CUR) base = cookie->position;
    else if (whence == SEEK_END) {
        struct stat st;
        if (fstat(cookie->fd, &st) != 0) return -1;
        base = st.st_size;
    } else return -1;
    if ((*offset < 0 && base < -*offset) ||
        (*offset > 0 && base > INT64_MAX - *offset)) return -1;
    cookie->position = base + *offset;
    *offset = cookie->position;
    return 0;
}

static int snapshot_fault_close(void *opaque) {
    snapshot_fault_cookie *cookie = opaque;
    return close(cookie->fd);
}

static uint16_t snapshot_half_bits(int global, int value, int layer, int head,
                                   int position, int dimension, int salt) {
    uint32_t x = (uint32_t)(position + 1) * UINT32_C(0x9e3779b1);
    x ^= (uint32_t)(layer + 3) * UINT32_C(0x85ebca6b);
    x ^= (uint32_t)(head + 5) * UINT32_C(0xc2b2ae35);
    x ^= (uint32_t)(dimension + 7) * UINT32_C(0x27d4eb2f);
    x ^= (uint32_t)(salt + 11) * UINT32_C(0x165667b1);
    x ^= (uint32_t)global << 29;
    x ^= (uint32_t)value << 30;
    x ^= x >> 16;
    return (uint16_t)x;
}

static _Float16 snapshot_half(int global, int value, int layer, int head,
                              int position, int dimension, int salt) {
    uint16_t bits = snapshot_half_bits(global, value, layer, head, position,
                                       dimension, salt);
    _Float16 value16;
    memcpy(&value16, &bits, sizeof value16);
    return value16;
}

static float snapshot_logit(int index, int salt) {
    return (float)((index * 17 + salt * 101) % 10007) / 37.0f - 100.0f;
}

static int32_t snapshot_token(int index, int salt) {
    return 2 + (index * 7919 + salt * 104729) % (XE_VOCAB - 2);
}

static void snapshot_fill(xe_session *s, int n, int salt) {
    s->n_tokens = n;
    for (int i = 0; i < n; i++) s->tokens[i] = snapshot_token(i, salt);
    for (int i = 0; i < XE_VOCAB; i++) s->logits[i] = snapshot_logit(i, salt);
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        int global = XE_IS_GLOBAL(layer);
        int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
        int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
        int capacity = global ? s->engine->context : XE_SWA_WINDOW;
        int first = global ? 0 : (n > XE_SWA_WINDOW ? n - XE_SWA_WINDOW : 0);
        for (int value = 0; value < 2; value++) {
            _Float16 *base = xe_kv_layer_ptr(s, layer, value);
            for (int head = 0; head < heads; head++) {
                for (int position = first; position < n; position++) {
                    int slot = global ? position : position & (XE_SWA_WINDOW - 1);
                    _Float16 *row = base +
                        ((size_t)head * capacity + slot) * dimension;
                    for (int d = 0; d < dimension; d++)
                        row[d] = snapshot_half(global, value, layer, head,
                                               position, d, salt);
                }
            }
        }
    }
}

static int snapshot_matches(const xe_session *s, int n, int salt) {
    if (s->n_tokens != n) return 0;
    for (int i = 0; i < n; i++)
        if (s->tokens[i] != snapshot_token(i, salt)) return 0;
    for (int i = 0; i < XE_VOCAB; i++)
        if (memcmp(&s->logits[i], &(float){snapshot_logit(i, salt)},
                   sizeof(float)) != 0) return 0;
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        int global = XE_IS_GLOBAL(layer);
        int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
        int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
        int capacity = global ? s->engine->context : XE_SWA_WINDOW;
        int first = global ? 0 : (n > XE_SWA_WINDOW ? n - XE_SWA_WINDOW : 0);
        for (int value = 0; value < 2; value++) {
            _Float16 *base = xe_kv_layer_ptr((xe_session *)s, layer, value);
            for (int head = 0; head < heads; head++) {
                for (int position = first; position < n; position++) {
                    int slot = global ? position : position & (XE_SWA_WINDOW - 1);
                    _Float16 *row = base +
                        ((size_t)head * capacity + slot) * dimension;
                    for (int d = 0; d < dimension; d++) {
                        _Float16 want = snapshot_half(global, value, layer,
                                                      head, position, d, salt);
                        if (memcmp(&row[d], &want, sizeof want) != 0) return 0;
                    }
                }
            }
        }
    }
    return 1;
}

static uint64_t snapshot_state_crc(xe_session *s) {
    uint64_t crc = 0;
    crc = format_crc64(crc, &s->n_tokens, sizeof s->n_tokens);
    crc = format_crc64(crc, s->tokens,
                       (size_t)s->n_tokens * sizeof(*s->tokens));
    crc = format_crc64(crc, s->logits, XE_VOCAB * sizeof(*s->logits));
    int n = s->n_tokens;
    for (int layer = 0; layer < XE_LAYERS; layer++) {
        int global = XE_IS_GLOBAL(layer);
        int heads = global ? XE_GLOBAL_KV_HEADS : XE_SWA_KV_HEADS;
        int dimension = global ? XE_GLOBAL_HEAD_DIM : XE_SWA_HEAD_DIM;
        int capacity = global ? s->engine->context : XE_SWA_WINDOW;
        int rows = global ? n : (n < XE_SWA_WINDOW ? n : XE_SWA_WINDOW);
        int first = global ? 0 : n - rows;
        for (int value = 0; value < 2; value++) {
            _Float16 *base = xe_kv_layer_ptr(s, layer, value);
            for (int head = 0; head < heads; head++) {
                _Float16 *head_base = base + (size_t)head * capacity * dimension;
                if (global) {
                    crc = format_crc64(crc, head_base,
                        (size_t)rows * dimension * sizeof(*head_base));
                } else {
                    int slot = first & (XE_SWA_WINDOW - 1);
                    int first_rows = rows < capacity - slot ? rows : capacity - slot;
                    crc = format_crc64(crc, head_base + (size_t)slot * dimension,
                        (size_t)first_rows * dimension * sizeof(*head_base));
                    crc = format_crc64(crc, head_base,
                        (size_t)(rows - first_rows) * dimension * sizeof(*head_base));
                }
            }
        }
    }
    return crc;
}

static int snapshot_file_half(FILE *file, uint64_t offset, uint16_t want) {
    uint8_t bytes[2];
    return format_seek(file, offset) && format_read(file, bytes, sizeof bytes) &&
           format_get_u16le(bytes) == want;
}

static int snapshot_logical_file(FILE *file, int n) {
    uint8_t directory[XE_SNAPSHOT_REQUIRED_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE];
    if (!format_seek(file, XE_SNAPSHOT_HEADER_SIZE) ||
        !format_read(file, directory, sizeof directory)) return 0;
    int rows = n < XE_SWA_WINDOW ? n : XE_SWA_WINDOW;
    int first = n - rows;
    uint64_t swa_offset = format_get_u64le(
        directory + (XE_SNAPSHOT_SWA_K - 1) * XE_SNAPSHOT_ENTRY_SIZE + 8);
    uint64_t global_offset = format_get_u64le(
        directory + (XE_SNAPSHOT_GLOBAL_V - 1) * XE_SNAPSHOT_ENTRY_SIZE + 8);
    uint64_t swa_last = swa_offset +
        ((uint64_t)(rows - 1) * XE_SWA_HEAD_DIM + 19) * sizeof(_Float16);
    uint64_t global_sample = global_offset +
        ((uint64_t)37 * XE_GLOBAL_HEAD_DIM + 23) * sizeof(_Float16);
    int ok = snapshot_file_half(file, swa_offset,
        snapshot_half_bits(0, 0, 0, 0, first, 0, 0));
    ok &= snapshot_file_half(file, swa_last,
        snapshot_half_bits(0, 0, 0, 0, n - 1, 19, 0));
    if (n > 37)
        ok &= snapshot_file_half(file, global_sample,
            snapshot_half_bits(1, 1, 5, 0, 37, 23, 0));
    return ok;
}

static int snapshot_round_trip(xe_session *s, int n) {
    snapshot_fill(s, n, 0);
    int32_t *expected_ids = malloc((size_t)n * sizeof(*expected_ids));
    if (!expected_ids) return 0;
    memcpy(expected_ids, s->tokens, (size_t)n * sizeof(*expected_ids));
    xe_tokens expected = { expected_ids, n, n };
    FILE *file = tmpfile();
    if (!file) {
        free(expected_ids);
        return 0;
    }
    uint64_t size = 0;
    int ok = xe_session_snapshot_size(s, &size) == XE_SNAPSHOT_OK;
    ok &= xe_session_snapshot_save(s, file) == XE_SNAPSHOT_OK;
    struct stat st;
    ok &= fstat(fileno(file), &st) == 0 && (uint64_t)st.st_size == size;
    ok &= snapshot_logical_file(file, n);
    snapshot_fill(s, n, 1);
    ok &= format_seek(file, 0);
    xe_snapshot_status status = xe_session_snapshot_load(s, file, &expected);
    ok &= status == XE_SNAPSHOT_OK;
    ok &= snapshot_matches(s, n, 0);
    fclose(file);
    free(expected_ids);
    printf("snapshot: round trip N=%d %s\n", n, ok ? "PASS" : "FAIL");
    return ok;
}

static int snapshot_rewrite_header(FILE *file, const uint8_t *header) {
    return format_seek(file, 0) &&
           format_write(file, header, XE_SNAPSHOT_HEADER_SIZE) &&
           fflush(file) == 0;
}

static int snapshot_rewrite_directory(FILE *file, uint8_t *header,
                                      const uint8_t *directory) {
    format_put_u64le(header + 88,
        format_crc64(0, directory,
            XE_SNAPSHOT_REQUIRED_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE));
    format_put_u64le(header + 376, format_crc64(0, header, 376));
    return snapshot_rewrite_header(file, header) &&
           format_seek(file, XE_SNAPSHOT_HEADER_SIZE) &&
           format_write(file, directory,
                        XE_SNAPSHOT_REQUIRED_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE) &&
           fflush(file) == 0;
}

static int snapshot_rejected_unchanged(xe_session *s, FILE *file,
                                       const xe_tokens *expected,
                                       xe_snapshot_status want,
                                       uint64_t before) {
    if (!format_seek(file, 0)) return 0;
    xe_snapshot_status got = xe_session_snapshot_load(s, file, expected);
    return got == want && snapshot_state_crc(s) == before;
}

static int snapshot_rejected_empty(xe_session *s, FILE *file,
                                   const xe_tokens *expected,
                                   xe_snapshot_status want) {
    if (!format_seek(file, 0)) return 0;
    xe_snapshot_status got = xe_session_snapshot_load(s, file, expected);
    return got == want && s->n_tokens == 0;
}

static int snapshot_rejections(xe_session *s) {
    const int n = 17;
    snapshot_fill(s, n, 0);
    int32_t expected_ids[n];
    memcpy(expected_ids, s->tokens, sizeof expected_ids);
    xe_tokens expected = { expected_ids, n, n };
    FILE *file = tmpfile();
    if (!file || xe_session_snapshot_save(s, file) != XE_SNAPSHOT_OK) {
        if (file) fclose(file);
        return 0;
    }
    uint8_t original_header[XE_SNAPSHOT_HEADER_SIZE];
    uint8_t original_directory[XE_SNAPSHOT_REQUIRED_SECTIONS *
                               XE_SNAPSHOT_ENTRY_SIZE];
    int ok = format_seek(file, 0) &&
             format_read(file, original_header, sizeof original_header) &&
             format_read(file, original_directory, sizeof original_directory);
    snapshot_fill(s, n, 7);
    uint64_t before = snapshot_state_crc(s);
    static const xe_snapshot_status mismatch[XE_SNAPSHOT_FP_COUNT] = {
        XE_SNAPSHOT_MODEL_MISMATCH, XE_SNAPSHOT_TOKENIZER_MISMATCH,
        XE_SNAPSHOT_TEMPLATE_MISMATCH, XE_SNAPSHOT_LAYOUT_MISMATCH,
        XE_SNAPSHOT_CONTEXT_MISMATCH, XE_SNAPSHOT_NUMERIC_MISMATCH
    };
    for (int i = 0; i < XE_SNAPSHOT_FP_COUNT; i++) {
        uint8_t header[XE_SNAPSHOT_HEADER_SIZE];
        memcpy(header, original_header, sizeof header);
        header[96 + 32 * i] ^= 1;
        format_put_u64le(header + 376, format_crc64(0, header, 376));
        ok &= snapshot_rewrite_header(file, header);
        ok &= snapshot_rejected_unchanged(s, file, &expected, mismatch[i],
                                          before);
    }

    ok &= snapshot_rewrite_header(file, original_header);
    int32_t wrong_ids[n];
    memcpy(wrong_ids, expected_ids, sizeof wrong_ids);
    wrong_ids[n / 2] ^= 1;
    xe_tokens wrong = { wrong_ids, n, n };
    ok &= snapshot_rejected_unchanged(s, file, &wrong,
                                      XE_SNAPSHOT_TOKEN_MISMATCH, before);
    xe_tokens shorter = { expected_ids, n - 1, n };
    ok &= snapshot_rejected_unchanged(s, file, &shorter,
                                      XE_SNAPSHOT_TOKEN_MISMATCH, before);

    uint8_t header[XE_SNAPSHOT_HEADER_SIZE];
    uint8_t directory[sizeof original_directory];
    memcpy(header, original_header, sizeof header);
    header[24] ^= 1;
    ok &= snapshot_rewrite_header(file, header);
    ok &= snapshot_rejected_unchanged(s, file, &expected,
                                      XE_SNAPSHOT_FORMAT, before);
    memcpy(header, original_header, sizeof header);
    memcpy(directory, original_directory, sizeof directory);
    format_put_u32le(directory, 99);
    ok &= snapshot_rewrite_directory(file, header, directory);
    ok &= snapshot_rejected_unchanged(s, file, &expected,
                                      XE_SNAPSHOT_VERSION, before);

    memcpy(header, original_header, sizeof header);
    format_put_u64le(header + 72, UINT64_MAX);
    format_put_u64le(header + 376, format_crc64(0, header, 376));
    ok &= snapshot_rewrite_header(file, header);
    ok &= snapshot_rejected_unchanged(s, file, &expected,
                                      XE_SNAPSHOT_FORMAT, before);

    memcpy(header, original_header, sizeof header);
    memcpy(directory, original_directory, sizeof directory);
    format_put_u64le(directory + 16, UINT64_MAX);
    ok &= snapshot_rewrite_directory(file, header, directory);
    ok &= snapshot_rejected_unchanged(s, file, &expected,
                                      XE_SNAPSHOT_FORMAT, before);

    memcpy(header, original_header, sizeof header);
    memcpy(directory, original_directory, sizeof directory);
    ok &= snapshot_rewrite_directory(file, header, directory);
    uint64_t single_pass_bytes = XE_SNAPSHOT_HEADER_SIZE +
        XE_SNAPSHOT_REQUIRED_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE;
    for (int i = 0; i < XE_SNAPSHOT_REQUIRED_SECTIONS; i++)
        single_pass_bytes += format_get_u64le(
            original_directory + i * XE_SNAPSHOT_ENTRY_SIZE + 16);
    snapshot_fault_cookie complete_cookie = {
        .fd = dup(fileno(file)),
        .fail_after = single_pass_bytes
    };
    cookie_io_functions_t io = {
        .read = snapshot_fault_read,
        .seek = snapshot_fault_seek,
        .close = snapshot_fault_close
    };
    FILE *complete = complete_cookie.fd >= 0
        ? fopencookie(&complete_cookie, "r", io) : NULL;
    if (complete) setvbuf(complete, NULL, _IONBF, 0);
    snapshot_fill(s, n, 7);
    xe_snapshot_status complete_status = complete
        ? xe_session_snapshot_load(s, complete, &expected) : XE_SNAPSHOT_IO;
    ok &= complete_status == XE_SNAPSHOT_OK &&
          complete_cookie.read_bytes == single_pass_bytes &&
          snapshot_matches(s, n, 0);
    if (complete) fclose(complete);
    snapshot_fill(s, n, 7);

    uint64_t logits_offset = format_get_u64le(
        directory + (XE_SNAPSHOT_LOGITS - 1) * XE_SNAPSHOT_ENTRY_SIZE + 8);
    uint8_t byte;
    ok &= format_seek(file, logits_offset) && format_read(file, &byte, 1);
    byte ^= 1;
    ok &= format_seek(file, logits_offset) && format_write(file, &byte, 1) &&
          fflush(file) == 0;
    ok &= snapshot_rejected_empty(s, file, &expected, XE_SNAPSHOT_FORMAT);
    byte ^= 1;
    ok &= format_seek(file, logits_offset) && format_write(file, &byte, 1) &&
          fflush(file) == 0;
    snapshot_fill(s, n, 7);

    ok &= snapshot_rewrite_directory(file, original_header, original_directory);
    uint64_t commit_prefix = XE_SNAPSHOT_HEADER_SIZE +
        XE_SNAPSHOT_REQUIRED_SECTIONS * XE_SNAPSHOT_ENTRY_SIZE +
        format_get_u64le(original_directory +
            (XE_SNAPSHOT_TOKENS - 1) * XE_SNAPSHOT_ENTRY_SIZE + 16) +
        format_get_u64le(original_directory +
            (XE_SNAPSHOT_LOGITS - 1) * XE_SNAPSHOT_ENTRY_SIZE + 16);
    snapshot_fault_cookie cookie = {
        .fd = dup(fileno(file)),
        .fail_after = commit_prefix + 128
    };
    FILE *fault = cookie.fd >= 0 ? fopencookie(&cookie, "r", io) : NULL;
    if (fault) setvbuf(fault, NULL, _IONBF, 0);
    snapshot_fill(s, n, 7);
    xe_snapshot_status commit_failure = fault
        ? xe_session_snapshot_load(s, fault, &expected) : XE_SNAPSHOT_IO;
    ok &= commit_failure == XE_SNAPSHOT_IO && s->n_tokens == 0;
    if (fault) fclose(fault);
    snapshot_fill(s, n, 7);

    ok &= snapshot_rewrite_directory(file, original_header, original_directory);
    uint64_t file_size = format_get_u64le(original_header + 32);
    ok &= ftruncate(fileno(file), (off_t)(file_size - 1)) == 0;
    ok &= snapshot_rejected_unchanged(s, file, &expected,
                                      XE_SNAPSHOT_FORMAT, before);
    fclose(file);
    printf("snapshot: metadata rejection and destructive payload %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int snapshot_large(xe_session *session) {
    const int n = session->engine->context;
    session->n_tokens = n;
    for (int i = 0; i < n; i++) session->tokens[i] = snapshot_token(i, 3);
    for (int i = 0; i < XE_VOCAB; i++)
        session->logits[i] = snapshot_logit(i, 3);
    memset(session->swa_k, 0x11,
           XE_SWA_SLAB_ELEMS * sizeof(*session->swa_k));
    memset(session->swa_v, 0x22,
           XE_SWA_SLAB_ELEMS * sizeof(*session->swa_v));
    memset(session->global_k, 0x33,
           xe_global_slab_elems(session->engine) * sizeof(*session->global_k));
    memset(session->global_v, 0x44,
           xe_global_slab_elems(session->engine) * sizeof(*session->global_v));
    int32_t *expected_ids = malloc((size_t)n * sizeof(*expected_ids));
    if (!expected_ids) return 0;
    memcpy(expected_ids, session->tokens,
           (size_t)n * sizeof(*expected_ids));
    xe_tokens expected = { expected_ids, n, n };
    FILE *file = tmpfile();
    uint64_t size = 0;
    int ok = file != NULL &&
             xe_session_snapshot_size(session, &size) == XE_SNAPSHOT_OK &&
             size > UINT32_MAX &&
             xe_session_snapshot_save(session, file) == XE_SNAPSHOT_OK;
    struct stat st;
    ok &= file && fstat(fileno(file), &st) == 0 &&
          (uint64_t)st.st_size == size;
    memset(session->swa_k, 0xaa,
           XE_SWA_SLAB_ELEMS * sizeof(*session->swa_k));
    memset(session->swa_v, 0xbb,
           XE_SWA_SLAB_ELEMS * sizeof(*session->swa_v));
    memset(session->global_k, 0xcc,
           xe_global_slab_elems(session->engine) * sizeof(*session->global_k));
    memset(session->global_v, 0xdd,
           xe_global_slab_elems(session->engine) * sizeof(*session->global_v));
    if (file) ok &= xe_session_snapshot_load(session, file, &expected) ==
                    XE_SNAPSHOT_OK;
    ok &= session->n_tokens == n &&
          memcmp(session->tokens, expected_ids,
                 (size_t)n * sizeof(*expected_ids)) == 0;
    ok &= ((uint8_t *)session->swa_k)[0] == 0x11 &&
          ((uint8_t *)session->swa_k)[XE_SWA_SLAB_ELEMS *
                                      sizeof(*session->swa_k) - 1] == 0x11;
    ok &= ((uint8_t *)session->swa_v)[0] == 0x22 &&
          ((uint8_t *)session->global_k)[0] == 0x33 &&
          ((uint8_t *)session->global_k)[xe_global_slab_elems(session->engine) *
                                         sizeof(*session->global_k) - 1] == 0x33;
    ok &= ((uint8_t *)session->global_v)[0] == 0x44 &&
          ((uint8_t *)session->global_v)[xe_global_slab_elems(session->engine) *
                                         sizeof(*session->global_v) - 1] == 0x44;
    if (file) fclose(file);
    free(expected_ids);
    printf("snapshot: large streaming %llu bytes %s\n",
           (unsigned long long)size, ok ? "PASS" : "FAIL");
    return ok;
}

static void snapshot_swa_slot(_Float16 *k, _Float16 *v, int slot,
                              unsigned char value) {
    size_t bytes = XE_SWA_HEAD_DIM * sizeof(*k);
    for (int layer = 0; layer < XE_LAYERS - XE_GLOBAL_LAYERS; layer++) {
        size_t layer_offset = (size_t)layer * XE_SWA_LAYER_ELEMS;
        for (int head = 0; head < XE_SWA_KV_HEADS; head++) {
            size_t offset = layer_offset +
                ((size_t)head * XE_SWA_WINDOW + slot) * XE_SWA_HEAD_DIM;
            memset(k + offset, value, bytes);
            memset(v + offset, value, bytes);
        }
    }
}

static int snapshot_twin_wrap(xe_session *source) {
    size_t bytes = XE_SWA_SLAB_ELEMS * sizeof(*source->swa_k);
    memset(source->swa_k, 0x11, bytes);
    memset(source->swa_v, 0x11, bytes);
    memset(source->swa_spare_k, 0x11, bytes);
    memset(source->swa_spare_v, 0x11, bytes);
    memset(source->swa_dirty, 0, sizeof source->swa_dirty);
    source->n_tokens = 1031;
    for (int pos = 0; pos < source->n_tokens; pos++)
        source->tokens[pos] = snapshot_token(pos, 11);
    xe_session *shadow = xe_session_shadow_new(source);
    if (!shadow) return 0;
    xe_session_shadow_reserve(shadow, 1057);
    for (int pos = 1031; pos < 1049; pos++) {
        snapshot_swa_slot(source->swa_k, source->swa_v,
                          pos & (XE_SWA_WINDOW - 1), 0x33);
        source->tokens[pos] = snapshot_token(pos, 13);
    }
    source->n_tokens = 1049;
    for (int pos = 1031; pos < 1057; pos++) {
        snapshot_swa_slot(shadow->swa_k, shadow->swa_v,
                          pos & (XE_SWA_WINDOW - 1), 0x55);
        shadow->tokens[pos] = snapshot_token(pos, 17);
    }
    shadow->n_tokens = 1057;
    memset(shadow->hidden, 0x66, XE_EMBD * sizeof(*shadow->hidden));
    memset(shadow->logits, 0x77, XE_VOCAB * sizeof(*shadow->logits));
    int promoted = xe_session_shadow_promote(source, shadow);
    xe_session_free(shadow);
    xe_session *next = xe_session_shadow_new(source);
    int equal = next && memcmp(source->swa_k, next->swa_k, bytes) == 0 &&
                memcmp(source->swa_v, next->swa_v, bytes) == 0;
    xe_session_free(next);
    int ok = promoted && source->n_tokens == 1057 && equal;
    printf("snapshot: twin SWA wrap promote/repair %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

static int snapshot_context_mismatch(xe_session *source) {
    snapshot_fill(source, 17, 0);
    xe_tokens expected = { source->tokens, 17, source->engine->context };
    FILE *file = tmpfile();
    if (!file || xe_session_snapshot_save(source, file) != XE_SNAPSHOT_OK) {
        if (file) fclose(file);
        return 0;
    }
    xe_engine other = {0};
    other.context = source->engine->context == 64 ? 129 : 64;
    xe_gpu_init(&other);
    xe_snapshot_fingerprints_init(&other);
    xe_session *target = xe_session_new(&other);
    snapshot_fill(target, 17, 7);
    uint64_t before = snapshot_state_crc(target);
    int ok = xe_context_size(source->engine) != xe_context_size(&other) &&
        snapshot_rejected_unchanged(target, file, &expected,
                                    XE_SNAPSHOT_CONTEXT_MISMATCH, before);
    xe_session_free(target);
    xe_worker_pool_destroy(&other);
    xe_gpu_destroy(&other);
    fclose(file);
    printf("snapshot: different runtime capacities rejected unchanged %s\n",
           ok ? "PASS" : "FAIL");
    return ok;
}

int main(int argc, char **argv) {
    xe_engine engine;
    memset(&engine, 0, sizeof engine);
    engine.context = XE_CONTEXT_DEFAULT;
    int run_large = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--large")) run_large = 1;
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc) {
            char *end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end ||
                value < XE_CONTEXT_MIN || value > XE_CONTEXT_MAX) return 2;
            engine.context = (int)value;
        } else return 2;
    }
    xe_gpu_init(&engine);
    xe_snapshot_fingerprints_init(&engine);
    xe_session *session = xe_session_new(&engine);
    int ok = snapshot_round_trip(session, 17);
    ok &= snapshot_round_trip(session, engine.context < 1057 ? engine.context : 1057);
    ok &= snapshot_context_mismatch(session);
    ok &= snapshot_rejections(session);
    if (engine.context >= 1057) ok &= snapshot_twin_wrap(session);
    session->n_tokens = session->engine->context;
    uint64_t maximum_size = 0;
    int large = xe_session_snapshot_size(session, &maximum_size) ==
                XE_SNAPSHOT_OK && maximum_size > 0 &&
                (engine.context != XE_CONTEXT_MAX || maximum_size > UINT32_MAX);
    ok &= large;
    printf("snapshot: capacity %d maximum size %llu %s\n",
           engine.context, (unsigned long long)maximum_size, large ? "PASS" : "FAIL");
    if (run_large)
        ok &= snapshot_large(session);
    xe_session_free(session);
    xe_worker_pool_destroy(&engine);
    xe_gpu_destroy(&engine);
    return ok ? 0 : 1;
}
