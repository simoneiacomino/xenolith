#define _GNU_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "conversation.h"
#include "format.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum {
    CONVERSATION_HEADER_SIZE = 64,
    CONVERSATION_FRAME_HEADER_SIZE = 32,
    CONVERSATION_SETTINGS_OLD_SIZE = 40,
    CONVERSATION_SETTINGS_SIZE = 56,
    CONVERSATION_FRAME_CRITICAL = 1,
    CONVERSATION_MAX_EVENT_TYPE = 12,
    CONVERSATION_TOKEN_LIMIT = 262143,
    CONVERSATION_CONTEXT_CAPACITY = 262144
};

#define CONVERSATION_FIELD_OPTIONAL UINT32_C(0x80000000)

static const uint8_t conversation_magic[8] = {
    'X', 'E', 'C', 'O', 'N', 'V', 'R', 'C'
};

typedef struct {
    uint8_t *data;
    size_t length;
    size_t capacity;
} conversation_buffer;

typedef struct {
    conversation_event view;
    uint8_t *payload;
    uint64_t payload_length;
    uint32_t flags;
    conversation_block *blocks;
    int32_t *token_copy;
    int32_t *alternate_token_copy;
    int32_t *raw_token_copy;
    char *server;
    char *tool;
} conversation_node;

typedef struct {
    uint64_t call_id;
    int has_result;
    uint64_t event_index;
    uint64_t result_index;
} conversation_call;

typedef struct {
    uint64_t index;
    uint64_t boundary;
} conversation_visible;

struct conversation_store {
    int sessions_fd;
    int trash_fd;
};

struct conversation {
    int fd;
    int readonly;
    int failed;
    conversation_id id;
    int64_t created;
    int64_t updated;
    int64_t last_timestamp;
    uint64_t valid_end;
    uint64_t append_end;
    conversation_node *events;
    size_t event_count;
    size_t event_capacity;
    int32_t *tokens;
    uint64_t token_count;
    uint64_t token_capacity;
    char *title;
    char *workspace;
    conversation_settings settings;
    int has_settings;
    uint64_t epoch;
    kvstore_id snapshot;
    uint64_t snapshot_boundary;
    int has_snapshot;
    uint64_t last_generation_id;
    int generation_pending;
    conversation_call *calls;
    size_t call_count;
    size_t call_capacity;
    conversation_visible *visible;
    size_t visible_count;
    size_t visible_capacity;
    uint64_t applied_events;
    uint64_t appended_tokens;
    uint8_t chain[32];
    format_sha256 chain_hasher;
    format_sha256 token_hasher;
    uint64_t uncommitted;
};

typedef struct {
    uint64_t last_commit_end;
    int64_t last_commit_time;
    uint8_t last_commit_chain[32];
    int has_commit;
    int damaged;
    int unknown_critical;
    uint64_t stop_offset;
} conversation_scan;

__attribute__((weak)) int conversation_fault(const char *point) {
    (void)point;
    return 0;
}

static int64_t conversation_now(void) {
    struct timespec now;
    if (clock_gettime(CLOCK_REALTIME, &now) != 0) return 0;
    return (int64_t)now.tv_sec * INT64_C(1000000000) + now.tv_nsec;
}

static int conversation_random(void *data, size_t length) {
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

static void conversation_id_hex(const conversation_id *id, char out[33]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 16; i++) {
        out[2 * i] = hex[id->bytes[i] >> 4];
        out[2 * i + 1] = hex[id->bytes[i] & 15];
    }
    out[32] = '\0';
}

static int conversation_hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static int conversation_id_from_hex(const char *text, conversation_id *id) {
    if (!text || strlen(text) != 32) return 0;
    for (int i = 0; i < 16; i++) {
        int high = conversation_hex_value(text[2 * i]);
        int low = conversation_hex_value(text[2 * i + 1]);
        if (high < 0 || low < 0) return 0;
        id->bytes[i] = (uint8_t)(high << 4 | low);
    }
    return 1;
}

static int conversation_utf8_valid(const uint8_t *p, size_t length) {
    size_t i = 0;
    while (i < length) {
        uint8_t c = p[i];
        size_t extra;
        uint32_t minimum, code;
        if (c < 0x80) {
            i++;
            continue;
        } else if ((c & 0xe0) == 0xc0) {
            extra = 1;
            minimum = 0x80;
            code = c & 0x1f;
        } else if ((c & 0xf0) == 0xe0) {
            extra = 2;
            minimum = 0x800;
            code = c & 0x0f;
        } else if ((c & 0xf8) == 0xf0) {
            extra = 3;
            minimum = 0x10000;
            code = c & 0x07;
        } else {
            return 0;
        }
        if (length - i <= extra) return 0;
        for (size_t j = 1; j <= extra; j++) {
            if ((p[i + j] & 0xc0) != 0x80) return 0;
            code = code << 6 | (p[i + j] & 0x3f);
        }
        if (code < minimum || code > 0x10ffff ||
            (code >= 0xd800 && code <= 0xdfff)) return 0;
        i += extra + 1;
    }
    return 1;
}

static int conversation_write_at(int fd, uint64_t offset, const void *data,
                                 size_t length) {
    const uint8_t *p = data;
    while (length) {
        ssize_t n = pwrite(fd, p, length, (off_t)offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        p += n;
        offset += (uint64_t)n;
        length -= (size_t)n;
    }
    return 1;
}

static int conversation_read_at(int fd, uint64_t offset, void *data,
                                size_t length) {
    uint8_t *p = data;
    while (length) {
        ssize_t n = pread(fd, p, length, (off_t)offset);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return 0;
        p += n;
        offset += (uint64_t)n;
        length -= (size_t)n;
    }
    return 1;
}

static int conversation_buffer_reserve(conversation_buffer *buffer,
                                       size_t extra) {
    if (extra > SIZE_MAX - buffer->length) return 0;
    size_t need = buffer->length + extra;
    if (need <= buffer->capacity) return 1;
    size_t capacity = buffer->capacity ? buffer->capacity : 256;
    while (capacity < need) {
        if (capacity > SIZE_MAX / 2) return 0;
        capacity *= 2;
    }
    uint8_t *grown = realloc(buffer->data, capacity);
    if (!grown) return 0;
    buffer->data = grown;
    buffer->capacity = capacity;
    return 1;
}

static int conversation_buffer_put(conversation_buffer *buffer,
                                   const void *data, size_t length) {
    if (!length) return 1;
    if (!conversation_buffer_reserve(buffer, length)) return 0;
    memcpy(buffer->data + buffer->length, data, length);
    buffer->length += length;
    return 1;
}

static int conversation_buffer_field(conversation_buffer *buffer,
                                     uint32_t field, const void *data,
                                     uint64_t length) {
    uint8_t header[16];
    format_put_u32le(header, field);
    format_put_u32le(header + 4, 0);
    format_put_u64le(header + 8, length);
    return conversation_buffer_put(buffer, header, sizeof header) &&
           conversation_buffer_put(buffer, data, (size_t)length);
}

static int conversation_buffer_field_u32(conversation_buffer *buffer,
                                         uint32_t field, uint32_t value) {
    uint8_t bytes[4];
    format_put_u32le(bytes, value);
    return conversation_buffer_field(buffer, field, bytes, sizeof bytes);
}

static int conversation_buffer_field_u64(conversation_buffer *buffer,
                                         uint32_t field, uint64_t value) {
    uint8_t bytes[8];
    format_put_u64le(bytes, value);
    return conversation_buffer_field(buffer, field, bytes, sizeof bytes);
}

static void conversation_settings_pack(const conversation_settings *settings,
                                       uint8_t out[CONVERSATION_SETTINGS_SIZE]) {
    uint32_t temperature, top_p;
    memcpy(&temperature, &settings->temperature, 4);
    memcpy(&top_p, &settings->top_p, 4);
    format_put_u32le(out, temperature);
    format_put_u32le(out + 4, (uint32_t)settings->top_k);
    format_put_u32le(out + 8, top_p);
    format_put_u32le(out + 12, (uint32_t)settings->max_tokens);
    format_put_u32le(out + 16, settings->sampler_abi);
    format_put_u32le(out + 20, 0);
    format_put_u64le(out + 24, settings->rng_seed);
    format_put_u64le(out + 32, settings->rng_state);
    format_put_u32le(out + 40, settings->reasoning_effort);
    format_put_u32le(out + 44, settings->reasoning_history);
    format_put_u32le(out + 48, (uint32_t)settings->reasoning_budget);
    format_put_u32le(out + 52, 0);
}

static int conversation_settings_unpack(
        const uint8_t *in, uint64_t length,
        conversation_settings *settings) {
    if (length != CONVERSATION_SETTINGS_OLD_SIZE &&
        length != CONVERSATION_SETTINGS_SIZE) return 0;
    uint32_t temperature = format_get_u32le(in);
    uint32_t top_p = format_get_u32le(in + 8);
    memcpy(&settings->temperature, &temperature, 4);
    settings->top_k = (int32_t)format_get_u32le(in + 4);
    memcpy(&settings->top_p, &top_p, 4);
    settings->max_tokens = (int32_t)format_get_u32le(in + 12);
    settings->sampler_abi = format_get_u32le(in + 16);
    settings->rng_seed = format_get_u64le(in + 24);
    settings->rng_state = format_get_u64le(in + 32);
    settings->reasoning_effort = CONVERSATION_REASONING_OFF;
    settings->reasoning_history = CONVERSATION_REASONING_DISCARD;
    settings->reasoning_budget = -1;
    if (length == CONVERSATION_SETTINGS_SIZE) {
        settings->reasoning_effort = format_get_u32le(in + 40);
        settings->reasoning_history = format_get_u32le(in + 44);
        settings->reasoning_budget = (int32_t)format_get_u32le(in + 48);
        if (format_get_u32le(in + 52) != 0 ||
            settings->reasoning_effort > CONVERSATION_REASONING_MAX ||
            settings->reasoning_history >
                CONVERSATION_REASONING_PRESERVE_TOOL_CALLS ||
            settings->reasoning_budget < -1)
            return 0;
    }
    return format_get_u32le(in + 20) == 0;
}

typedef struct {
    const uint8_t *data;
    uint64_t length;
    int present;
} conversation_field;

static int conversation_fields_parse(const uint8_t *payload, uint64_t length,
                                     conversation_field *fields,
                                     uint32_t field_count) {
    for (uint32_t i = 0; i < field_count; i++) {
        fields[i].data = NULL;
        fields[i].length = 0;
        fields[i].present = 0;
    }
    uint64_t offset = 0;
    while (offset < length) {
        if (length - offset < 16) return 0;
        uint32_t field = format_get_u32le(payload + offset);
        uint32_t reserved = format_get_u32le(payload + offset + 4);
        uint64_t field_length = format_get_u64le(payload + offset + 8);
        if (reserved != 0 || field_length > length - offset - 16) return 0;
        uint32_t index = field & ~CONVERSATION_FIELD_OPTIONAL;
        if (index >= 1 && index <= field_count) {
            if (fields[index - 1].present) return 0;
            fields[index - 1].data = payload + offset + 16;
            fields[index - 1].length = field_length;
            fields[index - 1].present = 1;
        } else if (!(field & CONVERSATION_FIELD_OPTIONAL)) {
            return 0;
        }
        offset += 16 + field_length;
    }
    return offset == length;
}

static int conversation_field_u32(const conversation_field *field,
                                  uint32_t *value) {
    if (!field->present || field->length != 4) return 0;
    *value = format_get_u32le(field->data);
    return 1;
}

static int conversation_field_u64(const conversation_field *field,
                                  uint64_t *value) {
    if (!field->present || field->length != 8) return 0;
    *value = format_get_u64le(field->data);
    return 1;
}

static int conversation_blocks_parse(const conversation_field *field,
                                     conversation_block **blocks,
                                     uint32_t *count) {
    *blocks = NULL;
    *count = 0;
    if (!field->present) return 0;
    const uint8_t *p = field->data;
    uint64_t remaining = field->length;
    uint32_t total = 0;
    while (remaining) {
        if (remaining < 16) return 0;
        uint64_t length = format_get_u64le(p + 8);
        if (format_get_u32le(p + 4) != 0 || length > remaining - 16) return 0;
        uint32_t block_format = format_get_u32le(p);
        if (block_format < CONVERSATION_BLOCK_TEXT ||
            block_format > CONVERSATION_BLOCK_BINARY) return 0;
        p += 16 + length;
        remaining -= 16 + length;
        total++;
    }
    if (!total) return 1;
    conversation_block *parsed = calloc(total, sizeof *parsed);
    if (!parsed) return -1;
    p = field->data;
    for (uint32_t i = 0; i < total; i++) {
        parsed[i].format = format_get_u32le(p);
        parsed[i].length = format_get_u64le(p + 8);
        parsed[i].data = p + 16;
        p += 16 + parsed[i].length;
    }
    *blocks = parsed;
    *count = total;
    return 1;
}

static int conversation_tokens_decode(conversation_node *node,
                                      const conversation_field *field,
                                      int variant) {
    if (!field->present || field->length % 4) return 0;
    uint32_t count = (uint32_t)(field->length / 4);
    if (variant == 1) node->view.alternate_token_count = count;
    else if (variant == 2) node->view.raw_token_count = count;
    else node->view.token_count = count;
    if (!count) return 1;
    int32_t **copy = variant == 1 ? &node->alternate_token_copy
                     : variant == 2 ? &node->raw_token_copy
                                    : &node->token_copy;
    *copy = malloc((size_t)count * sizeof(**copy));
    if (!*copy) return -1;
    for (uint32_t i = 0; i < count; i++) {
        int32_t token = (int32_t)format_get_u32le(field->data + 4 * i);
        if (token < 0 || token > CONVERSATION_TOKEN_LIMIT) return 0;
        (*copy)[i] = token;
    }
    if (variant == 1) node->view.alternate_tokens = *copy;
    else if (variant == 2) node->view.raw_tokens = *copy;
    else node->view.tokens = *copy;
    return 1;
}

static int conversation_node_parse(conversation_node *node) {
    conversation_event *view = &node->view;
    const uint8_t *payload = node->payload;
    uint64_t length = node->payload_length;
    conversation_field fields[12];
    switch (view->type) {
    case CONVERSATION_EVENT_TITLE:
    case CONVERSATION_EVENT_WORKSPACE:
        if (!conversation_fields_parse(payload, length, fields, 1) ||
            !fields[0].present) return 0;
        view->render = fields[0].data;
        view->render_length = fields[0].length;
        return 1;
    case CONVERSATION_EVENT_SETTINGS:
        if (!conversation_fields_parse(payload, length, fields, 1) ||
            !fields[0].present) return 0;
        return conversation_settings_unpack(fields[0].data,
                                            fields[0].length,
                                            &view->settings);
    case CONVERSATION_EVENT_MESSAGE: {
        if (!conversation_fields_parse(payload, length, fields, 8) ||
            !conversation_field_u32(&fields[0], &view->role) ||
            view->role < CONVERSATION_ROLE_SYSTEM ||
            view->role > CONVERSATION_ROLE_TOOL ||
            !fields[2].present) return 0;
        int blocks = conversation_blocks_parse(&fields[1], &node->blocks,
                                               &view->block_count);
        if (blocks <= 0) return blocks;
        view->blocks = node->blocks;
        view->render = fields[2].data;
        view->render_length = fields[2].length;
        int decoded = conversation_tokens_decode(node, &fields[3], 0);
        if (decoded <= 0) return decoded;
        if (fields[4].present != fields[5].present) return 0;
        if (fields[4].present) {
            view->alternate_render = fields[4].data;
            view->alternate_render_length = fields[4].length;
            decoded = conversation_tokens_decode(node, &fields[5], 1);
            if (decoded <= 0) return decoded;
        }
        if (fields[6].present) {
            view->reasoning = fields[6].data;
            view->reasoning_length = fields[6].length;
        }
        if (fields[7].present &&
            (!conversation_field_u32(&fields[7], &view->reasoning_close) ||
             view->reasoning_close > CONVERSATION_REASONING_EOS))
            return 0;
        return 1;
    }
    case CONVERSATION_EVENT_GENERATION_STARTED:
        if (!conversation_fields_parse(payload, length, fields, 2) ||
            !conversation_field_u64(&fields[0], &view->generation_id) ||
            !fields[1].present) return 0;
        return conversation_settings_unpack(fields[1].data,
                                            fields[1].length,
                                            &view->settings);
    case CONVERSATION_EVENT_GENERATION_RESULT: {
        if (!conversation_fields_parse(payload, length, fields, 12) ||
            !conversation_field_u64(&fields[0], &view->generation_id) ||
            !conversation_field_u32(&fields[1], &view->stop_reason) ||
            view->stop_reason < CONVERSATION_STOP_EOT_SAMPLED ||
            view->stop_reason > CONVERSATION_STOP_TOOL_CALLS ||
            !conversation_field_u64(&fields[2], &view->rng_after) ||
            !fields[4].present) return 0;
        int blocks = conversation_blocks_parse(&fields[3], &node->blocks,
                                               &view->block_count);
        if (blocks <= 0) return blocks;
        view->blocks = node->blocks;
        view->role = CONVERSATION_ROLE_ASSISTANT;
        view->render = fields[4].data;
        view->render_length = fields[4].length;
        int decoded = conversation_tokens_decode(node, &fields[5], 0);
        if (decoded <= 0) return decoded;
        if (fields[6].present != fields[7].present) return 0;
        if (fields[6].present) {
            view->alternate_render = fields[6].data;
            view->alternate_render_length = fields[6].length;
            decoded = conversation_tokens_decode(node, &fields[7], 1);
            if (decoded <= 0) return decoded;
        }
        if (fields[8].present) {
            view->reasoning = fields[8].data;
            view->reasoning_length = fields[8].length;
        }
        if (fields[9].present &&
            (!conversation_field_u32(&fields[9], &view->reasoning_close) ||
             view->reasoning_close > CONVERSATION_REASONING_EOS))
            return 0;
        if (fields[10].present != fields[11].present) return 0;
        if (fields[10].present) {
            view->raw_render = fields[10].data;
            view->raw_render_length = fields[10].length;
            decoded = conversation_tokens_decode(node, &fields[11], 2);
            if (decoded <= 0) return decoded;
        }
        return 1;
    }
    case CONVERSATION_EVENT_TOOL_STARTED:
        if (!conversation_fields_parse(payload, length, fields, 5) ||
            !conversation_field_u64(&fields[0], &view->call_id) ||
            !fields[1].present || !fields[2].present ||
            !fields[3].present || !fields[4].present ||
            fields[4].length != 32 ||
            !conversation_utf8_valid(fields[1].data, fields[1].length) ||
            !conversation_utf8_valid(fields[2].data, fields[2].length) ||
            memchr(fields[1].data, 0, fields[1].length) ||
            memchr(fields[2].data, 0, fields[2].length)) return 0;
        node->server = malloc(fields[1].length + 1);
        node->tool = malloc(fields[2].length + 1);
        if (!node->server || !node->tool) return -1;
        memcpy(node->server, fields[1].data, fields[1].length);
        node->server[fields[1].length] = '\0';
        memcpy(node->tool, fields[2].data, fields[2].length);
        node->tool[fields[2].length] = '\0';
        view->server = node->server;
        view->tool = node->tool;
        view->arguments = fields[3].data;
        view->arguments_length = fields[3].length;
        view->fingerprint = fields[4].data;
        return 1;
    case CONVERSATION_EVENT_TOOL_RESULT: {
        if (!conversation_fields_parse(payload, length, fields, 5) ||
            !conversation_field_u64(&fields[0], &view->call_id) ||
            !conversation_field_u32(&fields[1], &view->tool_status) ||
            view->tool_status < CONVERSATION_TOOL_OK ||
            view->tool_status > CONVERSATION_TOOL_ERROR ||
            !fields[3].present) return 0;
        int blocks = conversation_blocks_parse(&fields[2], &node->blocks,
                                               &view->block_count);
        if (blocks <= 0) return blocks;
        view->blocks = node->blocks;
        view->role = CONVERSATION_ROLE_TOOL;
        view->render = fields[3].data;
        view->render_length = fields[3].length;
        return conversation_tokens_decode(node, &fields[4], 0);
    }
    case CONVERSATION_EVENT_SNAPSHOT_REF:
        if (!conversation_fields_parse(payload, length, fields, 4) ||
            !fields[0].present || fields[0].length != 16 ||
            !conversation_field_u64(&fields[1], &view->snapshot_boundary) ||
            !conversation_field_u64(&fields[2], &view->epoch) ||
            !fields[3].present || fields[3].length != 8) return 0;
        memcpy(view->snapshot.bytes, fields[0].data, 16);
        return 1;
    case CONVERSATION_EVENT_CACHE_EPOCH:
        return conversation_fields_parse(payload, length, fields, 1) &&
               conversation_field_u64(&fields[0], &view->epoch);
    case CONVERSATION_EVENT_REWIND:
        return conversation_fields_parse(payload, length, fields, 1) &&
               conversation_field_u64(&fields[0], &view->rewind_target);
    default:
        return 0;
    }
}

static void conversation_node_free(conversation_node *node) {
    free(node->payload);
    free(node->blocks);
    free(node->token_copy);
    free(node->alternate_token_copy);
    free(node->raw_token_copy);
    free(node->server);
    free(node->tool);
}

static conversation_call *conversation_call_find(conversation *c,
                                                 uint64_t call_id) {
    for (size_t i = 0; i < c->call_count; i++)
        if (c->calls[i].call_id == call_id) return &c->calls[i];
    return NULL;
}

static conversation_status conversation_call_add(conversation *c,
                                                 uint64_t call_id,
                                                 uint64_t event_index) {
    if (c->call_count == c->call_capacity) {
        size_t capacity = c->call_capacity ? c->call_capacity * 2 : 8;
        conversation_call *grown = realloc(c->calls,
                                           capacity * sizeof(*grown));
        if (!grown) return CONVERSATION_NOMEM;
        c->calls = grown;
        c->call_capacity = capacity;
    }
    c->calls[c->call_count].call_id = call_id;
    c->calls[c->call_count].has_result = 0;
    c->calls[c->call_count].event_index = event_index;
    c->calls[c->call_count].result_index = 0;
    c->call_count++;
    return CONVERSATION_OK;
}

static conversation_status conversation_visible_push(conversation *c,
                                                     uint64_t index) {
    if (c->visible_count == c->visible_capacity) {
        size_t capacity = c->visible_capacity ? c->visible_capacity * 2 : 16;
        conversation_visible *grown = realloc(c->visible,
                                              capacity * sizeof(*grown));
        if (!grown) return CONVERSATION_NOMEM;
        c->visible = grown;
        c->visible_capacity = capacity;
    }
    c->visible[c->visible_count].index = index;
    c->visible[c->visible_count].boundary = c->token_count;
    c->visible_count++;
    return CONVERSATION_OK;
}

static int conversation_tokens_append(conversation *c, const int32_t *tokens,
                                      uint32_t count) {
    if (!count) return 1;
    if (c->token_count > UINT64_MAX - count ||
        c->token_count + count > SIZE_MAX / sizeof(*c->tokens)) return 0;
    uint64_t need = c->token_count + count;
    if (need > c->token_capacity) {
        uint64_t capacity = c->token_capacity ? c->token_capacity : 1024;
        while (capacity < need) capacity *= 2;
        int32_t *grown = realloc(c->tokens,
                                 (size_t)capacity * sizeof(*grown));
        if (!grown) return 0;
        c->tokens = grown;
        c->token_capacity = capacity;
    }
    memcpy(c->tokens + c->token_count, tokens,
           (size_t)count * sizeof(*tokens));
    c->token_count = need;
    c->appended_tokens += count;
    format_sha256_update(&c->token_hasher, tokens,
                         (size_t)count * sizeof(*tokens));
    return 1;
}

static conversation_status conversation_text_replace(char **slot,
                                                     const uint8_t *data,
                                                     uint64_t length) {
    char *copy = malloc((size_t)length + 1);
    if (!copy) return CONVERSATION_NOMEM;
    memcpy(copy, data, (size_t)length);
    copy[length] = '\0';
    free(*slot);
    *slot = copy;
    return CONVERSATION_OK;
}

static conversation_status conversation_rewind_apply(conversation *c,
                                                     uint64_t target) {
    if (target == CONVERSATION_REWIND_ALL) {
        c->token_count = 0;
        c->visible_count = 0;
        c->call_count = 0;
    } else {
        uint64_t position = c->visible_count;
        for (uint64_t i = 0; i < c->visible_count; i++)
            if (c->visible[i].index == target) {
                position = i;
                break;
            }
        if (position == c->visible_count) return CONVERSATION_FORMAT;
        c->token_count = c->visible[position].boundary;
        c->visible_count = position + 1;
        size_t kept = 0;
        for (size_t i = 0; i < c->call_count; i++) {
            if (c->calls[i].event_index > target) continue;
            if (c->calls[i].has_result &&
                c->calls[i].result_index > target) {
                c->calls[i].has_result = 0;
                c->calls[i].result_index = 0;
            }
            c->calls[kept++] = c->calls[i];
        }
        c->call_count = kept;
    }
    if (c->has_snapshot && c->snapshot_boundary > c->token_count) {
        c->has_snapshot = 0;
        c->snapshot_boundary = 0;
    }
    return CONVERSATION_OK;
}

static conversation_status conversation_apply(conversation *c,
                                              conversation_node *node) {
    conversation_event *view = &node->view;
    uint64_t index = c->applied_events;
    conversation_status status = CONVERSATION_FORMAT;
    switch (view->type) {
    case CONVERSATION_EVENT_TITLE:
        if (view->render_length > CONVERSATION_TITLE_MAX ||
            !conversation_utf8_valid(view->render, view->render_length) ||
            memchr(view->render, 0, view->render_length)) break;
        status = conversation_text_replace(&c->title, view->render,
                                           view->render_length);
        break;
    case CONVERSATION_EVENT_WORKSPACE:
        if (view->render_length > CONVERSATION_WORKSPACE_MAX ||
            view->render_length == 0 || view->render[0] != '/' ||
            memchr(view->render, 0, view->render_length)) break;
        status = conversation_text_replace(&c->workspace, view->render,
                                           view->render_length);
        break;
    case CONVERSATION_EVENT_SETTINGS:
        c->settings = view->settings;
        c->has_settings = 1;
        status = CONVERSATION_OK;
        break;
    case CONVERSATION_EVENT_MESSAGE:
    case CONVERSATION_EVENT_TOOL_RESULT:
        if (view->type == CONVERSATION_EVENT_TOOL_RESULT) {
            conversation_call *call = conversation_call_find(c,
                                                             view->call_id);
            if (!call || call->has_result) break;
            call->has_result = 1;
            call->result_index = index;
        }
        if (!conversation_tokens_append(c, view->tokens,
                                        view->token_count)) {
            status = CONVERSATION_NOMEM;
            break;
        }
        status = conversation_visible_push(c, index);
        break;
    case CONVERSATION_EVENT_GENERATION_STARTED:
        c->last_generation_id = view->generation_id;
        c->generation_pending = 1;
        status = CONVERSATION_OK;
        break;
    case CONVERSATION_EVENT_GENERATION_RESULT:
        if (c->generation_pending &&
            view->generation_id == c->last_generation_id)
            c->generation_pending = 0;
        if (c->has_settings) c->settings.rng_state = view->rng_after;
        if (!conversation_tokens_append(c, view->tokens,
                                        view->token_count)) {
            status = CONVERSATION_NOMEM;
            break;
        }
        status = conversation_visible_push(c, index);
        break;
    case CONVERSATION_EVENT_TOOL_STARTED:
        if (conversation_call_find(c, view->call_id)) break;
        status = conversation_call_add(c, view->call_id, index);
        break;
    case CONVERSATION_EVENT_SNAPSHOT_REF:
        if (view->epoch != c->epoch) break;
        if (view->snapshot_boundary > CONVERSATION_CONTEXT_CAPACITY) break;
        c->snapshot = view->snapshot;
        c->snapshot_boundary = view->snapshot_boundary;
        c->has_snapshot = 1;
        status = CONVERSATION_OK;
        break;
    case CONVERSATION_EVENT_CACHE_EPOCH:
        if (view->epoch != c->epoch + 1) break;
        c->epoch = view->epoch;
        c->has_snapshot = 0;
        c->snapshot_boundary = 0;
        status = CONVERSATION_OK;
        break;
    case CONVERSATION_EVENT_REWIND:
        if (view->rewind_target != CONVERSATION_REWIND_ALL &&
            view->rewind_target >= index) break;
        status = conversation_rewind_apply(c, view->rewind_target);
        break;
    default:
        break;
    }
    if (status == CONVERSATION_OK) c->applied_events++;
    return status;
}

static conversation_status conversation_node_store(conversation *c,
                                                   conversation_node *node) {
    if (c->event_count == c->event_capacity) {
        size_t capacity = c->event_capacity ? c->event_capacity * 2 : 16;
        conversation_node *grown = realloc(c->events,
                                           capacity * sizeof(*grown));
        if (!grown) return CONVERSATION_NOMEM;
        c->events = grown;
        c->event_capacity = capacity;
    }
    c->events[c->event_count++] = *node;
    return CONVERSATION_OK;
}

static int conversation_header_encode(const conversation_id *id,
                                      int64_t created,
                                      uint8_t out[CONVERSATION_HEADER_SIZE]) {
    memset(out, 0, CONVERSATION_HEADER_SIZE);
    memcpy(out, conversation_magic, sizeof conversation_magic);
    format_put_u32le(out + 8, 1);
    format_put_u32le(out + 12, CONVERSATION_HEADER_SIZE);
    memcpy(out + 16, id->bytes, 16);
    format_put_u64le(out + 32, (uint64_t)created);
    format_put_u64le(out + 56, format_crc64(0, out, 56));
    return 1;
}

static conversation_status conversation_header_decode(
        const uint8_t in[CONVERSATION_HEADER_SIZE], const conversation_id *id,
        int64_t *created) {
    if (memcmp(in, conversation_magic, sizeof conversation_magic) != 0)
        return CONVERSATION_FORMAT;
    if (format_get_u32le(in + 8) != 1 ||
        format_get_u32le(in + 12) != CONVERSATION_HEADER_SIZE)
        return CONVERSATION_VERSION;
    if (format_get_u64le(in + 56) != format_crc64(0, in, 56))
        return CONVERSATION_FORMAT;
    for (int i = 40; i < 56; i++)
        if (in[i]) return CONVERSATION_FORMAT;
    if (id && memcmp(in + 16, id->bytes, 16) != 0)
        return CONVERSATION_FORMAT;
    *created = (int64_t)format_get_u64le(in + 32);
    return CONVERSATION_OK;
}

static uint64_t conversation_frame_crc(const conversation_id *id,
                                       uint64_t offset,
                                       const uint8_t *header,
                                       const uint8_t *payload,
                                       uint64_t payload_length) {
    uint8_t seed[24];
    memcpy(seed, id->bytes, 16);
    format_put_u64le(seed + 16, offset);
    uint64_t crc = format_crc64(0, seed, sizeof seed);
    crc = format_crc64(crc, header, 24);
    return format_crc64(crc, payload, (size_t)payload_length);
}

static int conversation_frame_plausible(const uint8_t *header,
                                        uint64_t offset, uint64_t file_size) {
    uint32_t type = format_get_u32le(header);
    uint32_t flags = format_get_u32le(header + 4);
    uint64_t length = format_get_u64le(header + 16);
    return type >= 1 &&
           (flags & ~(uint32_t)CONVERSATION_FRAME_CRITICAL) == 0 &&
           length <= CONVERSATION_FRAME_MAX &&
           length <= file_size - offset - CONVERSATION_FRAME_HEADER_SIZE;
}

static int conversation_resync_finds_frame(const conversation_id *id, int fd,
                                           uint64_t start,
                                           uint64_t file_size) {
    if (file_size < CONVERSATION_FRAME_HEADER_SIZE ||
        start > file_size - CONVERSATION_FRAME_HEADER_SIZE) return 0;
    uint64_t remaining_size = file_size - start;
    if (remaining_size > SIZE_MAX / 2) return 1;
    uint8_t *remaining = malloc((size_t)remaining_size);
    if (!remaining) return 1;
    if (!conversation_read_at(fd, start, remaining, (size_t)remaining_size)) {
        free(remaining);
        return 1;
    }
    for (uint64_t o = 1; o + CONVERSATION_FRAME_HEADER_SIZE <= remaining_size;
         o++) {
        const uint8_t *header = remaining + o;
        if (!conversation_frame_plausible(header, start + o, file_size))
            continue;
        uint64_t length = format_get_u64le(header + 16);
        if (format_get_u64le(header + 24) ==
            conversation_frame_crc(id, start + o, header,
                                   header + CONVERSATION_FRAME_HEADER_SIZE,
                                   length)) {
            free(remaining);
            return 1;
        }
    }
    free(remaining);
    return 0;
}

static conversation_status conversation_scan_pass(
        conversation *c, uint64_t file_size, conversation_scan *scan,
        int build, uint64_t build_end) {
    uint64_t offset = CONVERSATION_HEADER_SIZE;
    uint8_t header[CONVERSATION_HEADER_SIZE];
    if (!conversation_read_at(c->fd, 0, header, sizeof header))
        return CONVERSATION_IO;
    int64_t created;
    conversation_status status = conversation_header_decode(
        header, &c->id, &created);
    if (status != CONVERSATION_OK) return status;
    c->created = created;

    format_sha256_bytes(header, CONVERSATION_HEADER_SIZE, c->chain);
    format_sha256_init(&c->chain_hasher);
    format_sha256_update(&c->chain_hasher, c->chain, 32);
    format_sha256_init(&c->token_hasher);
    uint64_t running_tokens = 0;
    uint64_t limit = build ? build_end : file_size;

    uint8_t *payload = NULL;
    size_t payload_capacity = 0;
    int pending_unknown = 0;
    while (offset < limit) {
        uint8_t frame[CONVERSATION_FRAME_HEADER_SIZE];
        if (limit - offset < CONVERSATION_FRAME_HEADER_SIZE ||
            !conversation_read_at(c->fd, offset, frame, sizeof frame)) break;
        uint32_t type = format_get_u32le(frame);
        uint32_t flags = format_get_u32le(frame + 4);
        int64_t timestamp = (int64_t)format_get_u64le(frame + 8);
        uint64_t payload_length = format_get_u64le(frame + 16);
        if (type < 1 ||
            (flags & ~(uint32_t)CONVERSATION_FRAME_CRITICAL) != 0 ||
            payload_length > CONVERSATION_FRAME_MAX ||
            payload_length > limit - offset - CONVERSATION_FRAME_HEADER_SIZE)
            break;
        if (payload_length > payload_capacity) {
            uint8_t *grown = realloc(payload, (size_t)payload_length);
            if (!grown) {
                free(payload);
                return CONVERSATION_NOMEM;
            }
            payload = grown;
            payload_capacity = (size_t)payload_length;
        }
        if (payload_length &&
            !conversation_read_at(c->fd,
                                  offset + CONVERSATION_FRAME_HEADER_SIZE,
                                  payload, (size_t)payload_length)) break;
        if (format_get_u64le(frame + 24) !=
            conversation_frame_crc(&c->id, offset, frame, payload,
                                   payload_length)) break;

        if (type == CONVERSATION_EVENT_COMMIT) {
            conversation_field fields[3];
            uint64_t commit_tokens;
            if (!conversation_fields_parse(payload, payload_length,
                                           fields, 3) ||
                !conversation_field_u64(&fields[0], &commit_tokens) ||
                !fields[1].present || fields[1].length != 32 ||
                !fields[2].present || fields[2].length != 32) break;
            format_sha256 chain_copy = c->chain_hasher;
            uint8_t chain_now[32];
            format_sha256_final(&chain_copy, chain_now);
            format_sha256 token_copy = c->token_hasher;
            uint8_t digest_now[32];
            format_sha256_final(&token_copy, digest_now);
            if (commit_tokens != running_tokens ||
                memcmp(fields[1].data, digest_now, 32) != 0 ||
                memcmp(fields[2].data, chain_now, 32) != 0) {
                scan->damaged = 1;
                break;
            }
            memcpy(c->chain, chain_now, 32);
            format_sha256_init(&c->chain_hasher);
            format_sha256_update(&c->chain_hasher, c->chain, 32);
            if (pending_unknown) scan->unknown_critical = 1;
            pending_unknown = 0;
            offset += CONVERSATION_FRAME_HEADER_SIZE + payload_length;
            scan->last_commit_end = offset;
            scan->last_commit_time = timestamp;
            memcpy(scan->last_commit_chain, c->chain, 32);
            scan->has_commit = 1;
            if (build) {
                c->updated = timestamp;
                c->last_timestamp = timestamp > c->last_timestamp
                                    ? timestamp : c->last_timestamp;
            }
            continue;
        }

        format_sha256_update(&c->chain_hasher, frame, sizeof frame);
        format_sha256_update(&c->chain_hasher, payload,
                             (size_t)payload_length);
        if (type > CONVERSATION_MAX_EVENT_TYPE) {
            if (flags & CONVERSATION_FRAME_CRITICAL) pending_unknown = 1;
            offset += CONVERSATION_FRAME_HEADER_SIZE + payload_length;
            continue;
        }

        conversation_node node;
        memset(&node, 0, sizeof node);
        node.view.type = type;
        node.view.timestamp = timestamp;
        node.flags = flags;
        node.payload_length = payload_length;
        node.payload = malloc(payload_length ? (size_t)payload_length : 1);
        if (!node.payload) {
            free(payload);
            return CONVERSATION_NOMEM;
        }
        memcpy(node.payload, payload, (size_t)payload_length);
        int parsed = conversation_node_parse(&node);
        conversation_status applied = CONVERSATION_FORMAT;
        if (parsed < 0) applied = CONVERSATION_NOMEM;
        else if (parsed) applied = conversation_apply(c, &node);
        if (applied == CONVERSATION_NOMEM) {
            conversation_node_free(&node);
            free(payload);
            return CONVERSATION_NOMEM;
        }
        if (applied != CONVERSATION_OK) {
            conversation_node_free(&node);
            break;
        }
        running_tokens = c->appended_tokens;
        if (build) {
            conversation_status stored = conversation_node_store(c, &node);
            if (stored != CONVERSATION_OK) {
                conversation_node_free(&node);
                free(payload);
                return stored;
            }
            c->last_timestamp = timestamp > c->last_timestamp
                                ? timestamp : c->last_timestamp;
        } else {
            conversation_node_free(&node);
        }
        offset += CONVERSATION_FRAME_HEADER_SIZE + payload_length;
    }
    free(payload);
    scan->stop_offset = offset;
    return CONVERSATION_OK;
}

static void conversation_state_reset(conversation *c) {
    for (size_t i = 0; i < c->event_count; i++)
        conversation_node_free(&c->events[i]);
    free(c->events);
    free(c->tokens);
    free(c->title);
    free(c->workspace);
    free(c->calls);
    free(c->visible);
    c->events = NULL;
    c->event_count = 0;
    c->event_capacity = 0;
    c->tokens = NULL;
    c->token_count = 0;
    c->token_capacity = 0;
    c->title = NULL;
    c->workspace = NULL;
    c->calls = NULL;
    c->call_count = 0;
    c->call_capacity = 0;
    c->visible = NULL;
    c->visible_count = 0;
    c->visible_capacity = 0;
    c->applied_events = 0;
    c->appended_tokens = 0;
    c->has_settings = 0;
    c->epoch = 0;
    c->has_snapshot = 0;
    c->snapshot_boundary = 0;
    c->generation_pending = 0;
    c->last_generation_id = 0;
    c->uncommitted = 0;
}

static conversation_status conversation_load(conversation *c,
                                             int *resumable) {
    struct stat st;
    if (fstat(c->fd, &st) != 0 || !S_ISREG(st.st_mode))
        return CONVERSATION_IO;
    uint64_t file_size = (uint64_t)st.st_size;
    if (file_size < CONVERSATION_HEADER_SIZE) return CONVERSATION_FORMAT;

    conversation_scan scan;
    memset(&scan, 0, sizeof scan);
    conversation_status status = conversation_scan_pass(c, file_size, &scan,
                                                        0, 0);
    if (status != CONVERSATION_OK) return status;
    int damaged = scan.damaged;
    if (!damaged && scan.stop_offset < file_size &&
        conversation_resync_finds_frame(&c->id, c->fd, scan.stop_offset,
                                        file_size))
        damaged = 1;
    uint64_t valid_end = scan.has_commit ? scan.last_commit_end
                                         : CONVERSATION_HEADER_SIZE;

    conversation_state_reset(c);
    conversation_scan rebuild;
    memset(&rebuild, 0, sizeof rebuild);
    status = conversation_scan_pass(c, file_size, &rebuild, 1, valid_end);
    if (status != CONVERSATION_OK) return status;
    if (rebuild.stop_offset != valid_end || rebuild.damaged) damaged = 1;

    c->valid_end = valid_end;
    c->append_end = valid_end;
    if (!rebuild.has_commit) c->updated = c->created;
    if (c->last_timestamp < c->updated) c->last_timestamp = c->updated;
    if (c->last_timestamp < c->created) c->last_timestamp = c->created;
    c->uncommitted = 0;
    int had_snapshot = c->has_snapshot;
    kvstore_id snapshot = c->snapshot;
    uint64_t snapshot_boundary = c->snapshot_boundary;
    status = conversation_project(
        c, c->has_settings && c->settings.reasoning_effort !=
                              CONVERSATION_REASONING_OFF,
        c->has_settings ? c->settings.reasoning_history
                        : CONVERSATION_REASONING_DISCARD);
    if (status != CONVERSATION_OK) return status;
    if (had_snapshot && snapshot_boundary <= c->token_count) {
        c->snapshot = snapshot;
        c->snapshot_boundary = snapshot_boundary;
        c->has_snapshot = 1;
    }
    *resumable = !damaged && !scan.unknown_critical;
    return damaged ? CONVERSATION_DAMAGED
                   : scan.unknown_critical ? CONVERSATION_VERSION
                                           : CONVERSATION_OK;
}

static int conversation_dirs_prepare(const char *state_dir, int *state_fd) {
    char *copy = strdup(state_dir);
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
    if (mkdir(copy, 0700) != 0 && errno != EEXIST) {
        free(copy);
        return 0;
    }
    int fd = open(copy, O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    free(copy);
    if (fd < 0) return 0;
    *state_fd = fd;
    return 1;
}

int conversation_default_state_dir(char *out, size_t cap) {
    const char *state = getenv("XDG_STATE_HOME");
    int n;
    if (state && *state == '/') {
        n = snprintf(out, cap, "%s/xenolith", state);
    } else {
        const char *home = getenv("HOME");
        if (!home || *home != '/') return 0;
        n = snprintf(out, cap, "%s/.local/state/xenolith", home);
    }
    return n > 0 && (size_t)n < cap;
}

conversation_status conversation_store_open(conversation_store **out,
                                            const char *state_dir) {
    if (!out || !state_dir || *state_dir != '/')
        return CONVERSATION_INVALID_ARGUMENT;
    *out = NULL;
    int state_fd;
    if (!conversation_dirs_prepare(state_dir, &state_fd))
        return CONVERSATION_IO;
    if (mkdirat(state_fd, "sessions", 0700) != 0 && errno != EEXIST) {
        close(state_fd);
        return CONVERSATION_IO;
    }
    if (mkdirat(state_fd, "trash", 0700) != 0 && errno != EEXIST) {
        close(state_fd);
        return CONVERSATION_IO;
    }
    int sessions_fd = openat(state_fd, "sessions",
                             O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    int trash_fd = openat(state_fd, "trash",
                          O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    close(state_fd);
    if (sessions_fd < 0 || trash_fd < 0) {
        if (sessions_fd >= 0) close(sessions_fd);
        if (trash_fd >= 0) close(trash_fd);
        return CONVERSATION_IO;
    }
    conversation_store *store = calloc(1, sizeof *store);
    if (!store) {
        close(sessions_fd);
        close(trash_fd);
        return CONVERSATION_NOMEM;
    }
    store->sessions_fd = sessions_fd;
    store->trash_fd = trash_fd;
    *out = store;
    return CONVERSATION_OK;
}

void conversation_store_close(conversation_store *store) {
    if (!store) return;
    close(store->sessions_fd);
    close(store->trash_fd);
    free(store);
}

static conversation *conversation_new(int fd, int readonly) {
    conversation *c = calloc(1, sizeof *c);
    if (!c) return NULL;
    c->fd = fd;
    c->readonly = readonly;
    return c;
}

void conversation_close(conversation *c) {
    if (!c) return;
    conversation_state_reset(c);
    close(c->fd);
    free(c);
}

conversation_status conversation_create(conversation_store *store,
                                        conversation **out,
                                        conversation_id *id) {
    if (!store || !out || !id) return CONVERSATION_INVALID_ARGUMENT;
    *out = NULL;
    char hex[33];
    int session_fd = -1;
    for (int attempt = 0; attempt < 32 && session_fd < 0; attempt++) {
        if (!conversation_random(id->bytes, sizeof id->bytes))
            return CONVERSATION_IO;
        conversation_id_hex(id, hex);
        if (mkdirat(store->sessions_fd, hex, 0700) != 0) {
            if (errno == EEXIST) continue;
            return CONVERSATION_IO;
        }
        session_fd = openat(store->sessions_fd, hex,
                            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (session_fd < 0) return CONVERSATION_IO;
    }
    if (session_fd < 0) return CONVERSATION_IO;

    int fd = openat(session_fd, "record.xcr",
                    O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600);
    if (fd < 0) {
        close(session_fd);
        return CONVERSATION_IO;
    }
    int64_t created = conversation_now();
    uint8_t header[CONVERSATION_HEADER_SIZE];
    conversation_header_encode(id, created, header);
    int ok = conversation_write_at(fd, 0, header, sizeof header) &&
             fsync(fd) == 0 && fsync(session_fd) == 0 &&
             fsync(store->sessions_fd) == 0 &&
             flock(fd, LOCK_EX | LOCK_NB) == 0;
    close(session_fd);
    if (!ok) {
        close(fd);
        int cleanup_fd = openat(store->sessions_fd, hex,
                                O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                O_NOFOLLOW);
        if (cleanup_fd >= 0) {
            unlinkat(cleanup_fd, "record.xcr", 0);
            close(cleanup_fd);
        }
        unlinkat(store->sessions_fd, hex, AT_REMOVEDIR);
        return CONVERSATION_IO;
    }
    conversation *c = conversation_new(fd, 0);
    if (!c) {
        close(fd);
        return CONVERSATION_NOMEM;
    }
    c->id = *id;
    c->created = created;
    c->updated = created;
    c->last_timestamp = created;
    c->valid_end = CONVERSATION_HEADER_SIZE;
    c->append_end = CONVERSATION_HEADER_SIZE;
    format_sha256_bytes(header, CONVERSATION_HEADER_SIZE, c->chain);
    format_sha256_init(&c->chain_hasher);
    format_sha256_update(&c->chain_hasher, c->chain, 32);
    format_sha256_init(&c->token_hasher);
    *out = c;
    return CONVERSATION_OK;
}

static conversation_status conversation_open_mode(conversation_store *store,
                                                  const conversation_id *id,
                                                  conversation **out,
                                                  int readonly,
                                                  int *resumable) {
    *out = NULL;
    char hex[33];
    conversation_id_hex(id, hex);
    int session_fd = openat(store->sessions_fd, hex,
                            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (session_fd < 0)
        return errno == ENOENT ? CONVERSATION_MISS : CONVERSATION_IO;
    int fd = openat(session_fd, "record.xcr",
                    (readonly ? O_RDONLY : O_RDWR) | O_CLOEXEC | O_NOFOLLOW);
    close(session_fd);
    if (fd < 0)
        return errno == ENOENT ? CONVERSATION_MISS : CONVERSATION_IO;
    if (!readonly && flock(fd, LOCK_EX | LOCK_NB) != 0) {
        close(fd);
        return errno == EWOULDBLOCK ? CONVERSATION_LOCKED : CONVERSATION_IO;
    }
    conversation *c = conversation_new(fd, readonly);
    if (!c) {
        close(fd);
        return CONVERSATION_NOMEM;
    }
    c->id = *id;
    conversation_status status = conversation_load(c, resumable);
    if (status != CONVERSATION_OK &&
        !(readonly && (status == CONVERSATION_DAMAGED ||
                       status == CONVERSATION_VERSION))) {
        conversation_close(c);
        return status;
    }
    if (!readonly) {
        struct stat st;
        if (fstat(c->fd, &st) != 0 ||
            ((uint64_t)st.st_size > c->valid_end &&
             (ftruncate(c->fd, (off_t)c->valid_end) != 0 ||
              fsync(c->fd) != 0))) {
            conversation_close(c);
            return CONVERSATION_IO;
        }
    }
    *out = c;
    return status;
}

conversation_status conversation_open(conversation_store *store,
                                      const conversation_id *id,
                                      conversation **out) {
    if (!store || !id || !out) return CONVERSATION_INVALID_ARGUMENT;
    int resumable = 0;
    return conversation_open_mode(store, id, out, 0, &resumable);
}

conversation_status conversation_rollback(conversation *c) {
    if (!c || c->readonly) return CONVERSATION_INVALID_ARGUMENT;
    if (ftruncate(c->fd, (off_t)c->valid_end) != 0 || fsync(c->fd) != 0) {
        c->failed = 1;
        return CONVERSATION_IO;
    }
    conversation_state_reset(c);
    c->failed = 0;
    int resumable = 0;
    conversation_status status = conversation_load(c, &resumable);
    if (status != CONVERSATION_OK) c->failed = 1;
    return status;
}

conversation_status conversation_delete(conversation_store *store,
                                        const conversation_id *id) {
    if (!store || !id) return CONVERSATION_INVALID_ARGUMENT;
    char hex[33], trash_name[64];
    conversation_id_hex(id, hex);
    conversation_id nonce;
    if (!conversation_random(nonce.bytes, sizeof nonce.bytes))
        return CONVERSATION_IO;
    char nonce_hex[33];
    conversation_id_hex(&nonce, nonce_hex);
    int n = snprintf(trash_name, sizeof trash_name, "%s-%.16s", hex,
                     nonce_hex);
    if (n < 0 || (size_t)n >= sizeof trash_name) return CONVERSATION_IO;
    if (renameat(store->sessions_fd, hex, store->trash_fd, trash_name) != 0)
        return errno == ENOENT ? CONVERSATION_MISS : CONVERSATION_IO;
    if (fsync(store->sessions_fd) != 0 || fsync(store->trash_fd) != 0)
        return CONVERSATION_IO;
    int trashed_fd = openat(store->trash_fd, trash_name,
                            O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (trashed_fd >= 0) {
        unlinkat(trashed_fd, "record.xcr", 0);
        close(trashed_fd);
        unlinkat(store->trash_fd, trash_name, AT_REMOVEDIR);
        fsync(store->trash_fd);
    }
    return CONVERSATION_OK;
}

conversation_status conversation_list(conversation_store *store,
                                      conversation_summary **out,
                                      size_t *count) {
    if (!store || !out || !count) return CONVERSATION_INVALID_ARGUMENT;
    *out = NULL;
    *count = 0;
    int duplicate = openat(store->sessions_fd, ".",
                           O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (duplicate < 0) return CONVERSATION_IO;
    DIR *directory = fdopendir(duplicate);
    if (!directory) {
        close(duplicate);
        return CONVERSATION_IO;
    }
    conversation_summary *summaries = NULL;
    size_t capacity = 0, total = 0;
    struct dirent *entry;
    conversation_status status = CONVERSATION_OK;
    while ((entry = readdir(directory)) != NULL) {
        conversation_id id;
        if (!conversation_id_from_hex(entry->d_name, &id)) continue;
        conversation *c = NULL;
        int resumable = 0;
        conversation_status opened = conversation_open_mode(
            store, &id, &c, 1, &resumable);
        if (opened != CONVERSATION_OK && !c) continue;
        if (total == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            conversation_summary *grown = realloc(
                summaries, next * sizeof(*grown));
            if (!grown) {
                conversation_close(c);
                status = CONVERSATION_NOMEM;
                break;
            }
            summaries = grown;
            capacity = next;
        }
        conversation_summary *summary = &summaries[total++];
        memset(summary, 0, sizeof *summary);
        summary->id = id;
        summary->created = c->created;
        summary->updated = c->updated;
        summary->token_count = c->token_count;
        summary->resumable = resumable;
        if (c->title) {
            strncpy(summary->title, c->title, sizeof summary->title - 1);
            summary->title[sizeof summary->title - 1] = '\0';
        }
        conversation_close(c);
    }
    closedir(directory);
    if (status != CONVERSATION_OK) {
        free(summaries);
        return status;
    }
    *out = summaries;
    *count = total;
    return CONVERSATION_OK;
}

static int conversation_discard_tail(conversation *c) {
    return ftruncate(c->fd, (off_t)c->append_end) == 0 &&
           fsync(c->fd) == 0;
}

static conversation_status conversation_frame_write(
        conversation *c, uint32_t type, uint32_t flags,
        const conversation_buffer *payload) {
    if (c->readonly || c->failed) return CONVERSATION_INVALID_ARGUMENT;
    if (payload->length > CONVERSATION_FRAME_MAX) return CONVERSATION_LIMIT;
    int64_t timestamp = conversation_now();
    if (timestamp < c->last_timestamp) timestamp = c->last_timestamp;

    uint8_t header[CONVERSATION_FRAME_HEADER_SIZE];
    format_put_u32le(header, type);
    format_put_u32le(header + 4, flags);
    format_put_u64le(header + 8, (uint64_t)timestamp);
    format_put_u64le(header + 16, payload->length);
    format_put_u64le(header + 24,
                     conversation_frame_crc(&c->id, c->append_end, header,
                                            payload->data, payload->length));

    if (!conversation_write_at(c->fd, c->append_end, header, sizeof header) ||
        (payload->length &&
         !conversation_write_at(c->fd,
                                c->append_end + sizeof header,
                                payload->data, payload->length)) ||
        conversation_fault("frame-written")) {
        if (!conversation_discard_tail(c)) c->failed = 1;
        return CONVERSATION_IO;
    }

    if (type != CONVERSATION_EVENT_COMMIT) {
        conversation_node node;
        memset(&node, 0, sizeof node);
        node.view.type = type;
        node.view.timestamp = timestamp;
        node.flags = flags;
        node.payload_length = payload->length;
        node.payload = malloc(payload->length ? payload->length : 1);
        if (!node.payload) {
            if (!conversation_discard_tail(c)) c->failed = 1;
            return CONVERSATION_NOMEM;
        }
        memcpy(node.payload, payload->data, payload->length);
        int parsed = conversation_node_parse(&node);
        conversation_status applied = CONVERSATION_INVALID_ARGUMENT;
        if (parsed < 0) applied = CONVERSATION_NOMEM;
        else if (parsed > 0) applied = conversation_apply(c, &node);
        if (applied == CONVERSATION_OK)
            applied = conversation_node_store(c, &node);
        if (applied != CONVERSATION_OK) {
            conversation_node_free(&node);
            if (applied == CONVERSATION_NOMEM) c->failed = 1;
            if (!conversation_discard_tail(c)) c->failed = 1;
            return applied;
        }
        format_sha256_update(&c->chain_hasher, header, sizeof header);
        format_sha256_update(&c->chain_hasher, payload->data,
                             payload->length);
        c->uncommitted++;
    }
    c->append_end += sizeof header + payload->length;
    c->last_timestamp = timestamp;
    return CONVERSATION_OK;
}

conversation_status conversation_commit(conversation *c) {
    if (!c) return CONVERSATION_INVALID_ARGUMENT;
    if (c->readonly || c->failed) return CONVERSATION_INVALID_ARGUMENT;
    format_sha256 chain_copy = c->chain_hasher;
    uint8_t chain_now[32];
    format_sha256_final(&chain_copy, chain_now);
    format_sha256 token_copy = c->token_hasher;
    uint8_t digest_now[32];
    format_sha256_final(&token_copy, digest_now);

    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_IO;
    if (conversation_buffer_field_u64(&payload, 1, c->appended_tokens) &&
        conversation_buffer_field(&payload, 2, digest_now, 32) &&
        conversation_buffer_field(&payload, 3, chain_now, 32)) {
        status = conversation_frame_write(c, CONVERSATION_EVENT_COMMIT,
                                          CONVERSATION_FRAME_CRITICAL,
                                          &payload);
    } else {
        status = CONVERSATION_NOMEM;
    }
    free(payload.data);
    if (status != CONVERSATION_OK) return status;
    if (conversation_fault("commit-synced") || fsync(c->fd) != 0) {
        c->failed = 1;
        return CONVERSATION_IO;
    }
    memcpy(c->chain, chain_now, 32);
    format_sha256_init(&c->chain_hasher);
    format_sha256_update(&c->chain_hasher, c->chain, 32);
    c->valid_end = c->append_end;
    c->updated = c->last_timestamp;
    c->uncommitted = 0;
    return CONVERSATION_OK;
}

conversation_status conversation_append_title(conversation *c,
                                              const char *title) {
    if (!c || !title) return CONVERSATION_INVALID_ARGUMENT;
    size_t length = strlen(title);
    if (length > CONVERSATION_TITLE_MAX ||
        !conversation_utf8_valid((const uint8_t *)title, length))
        return CONVERSATION_LIMIT;
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    if (conversation_buffer_field(&payload, 1, title, length))
        status = conversation_frame_write(c, CONVERSATION_EVENT_TITLE, 0,
                                          &payload);
    free(payload.data);
    return status;
}

conversation_status conversation_append_workspace(conversation *c,
                                                  const char *path) {
    if (!c || !path) return CONVERSATION_INVALID_ARGUMENT;
    size_t length = strlen(path);
    if (length == 0 || length > CONVERSATION_WORKSPACE_MAX || path[0] != '/')
        return CONVERSATION_LIMIT;
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    if (conversation_buffer_field(&payload, 1, path, length))
        status = conversation_frame_write(c, CONVERSATION_EVENT_WORKSPACE, 0,
                                          &payload);
    free(payload.data);
    return status;
}

conversation_status conversation_append_settings(
        conversation *c, const conversation_settings *settings) {
    if (!c || !settings ||
        settings->reasoning_effort > CONVERSATION_REASONING_MAX ||
        settings->reasoning_history >
            CONVERSATION_REASONING_PRESERVE_TOOL_CALLS ||
        settings->reasoning_budget < -1)
        return CONVERSATION_INVALID_ARGUMENT;
    uint8_t packed[CONVERSATION_SETTINGS_SIZE];
    conversation_settings_pack(settings, packed);
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    if (conversation_buffer_field(&payload, 1, packed, sizeof packed))
        status = conversation_frame_write(c, CONVERSATION_EVENT_SETTINGS,
                                          CONVERSATION_FRAME_CRITICAL,
                                          &payload);
    free(payload.data);
    return status;
}

static int conversation_buffer_blocks(conversation_buffer *payload,
                                      uint32_t field,
                                      const conversation_block *blocks,
                                      uint32_t block_count) {
    conversation_buffer packed = {0};
    for (uint32_t i = 0; i < block_count; i++) {
        uint8_t header[16];
        format_put_u32le(header, blocks[i].format);
        format_put_u32le(header + 4, 0);
        format_put_u64le(header + 8, blocks[i].length);
        if (!conversation_buffer_put(&packed, header, sizeof header) ||
            !conversation_buffer_put(&packed, blocks[i].data,
                                     (size_t)blocks[i].length)) {
            free(packed.data);
            return 0;
        }
    }
    int ok = conversation_buffer_field(payload, field, packed.data,
                                       packed.length);
    free(packed.data);
    return ok;
}

static int conversation_buffer_tokens(conversation_buffer *payload,
                                      uint32_t field, const int32_t *tokens,
                                      uint32_t token_count) {
    conversation_buffer packed = {0};
    for (uint32_t i = 0; i < token_count; i++) {
        uint8_t bytes[4];
        format_put_u32le(bytes, (uint32_t)tokens[i]);
        if (!conversation_buffer_put(&packed, bytes, sizeof bytes)) {
            free(packed.data);
            return 0;
        }
    }
    int ok = conversation_buffer_field(payload, field, packed.data,
                                       packed.length);
    free(packed.data);
    return ok;
}

conversation_status conversation_append_message(
        conversation *c, uint32_t role,
        const conversation_block *blocks, uint32_t block_count,
        const void *render, uint64_t render_length,
        const int32_t *tokens, uint32_t token_count) {
    return conversation_append_message_variants(
        c, role, blocks, block_count, render, render_length,
        tokens, token_count, NULL, 0, NULL, 0, NULL, 0,
        CONVERSATION_REASONING_NONE);
}

conversation_status conversation_append_message_variants(
        conversation *c, uint32_t role,
        const conversation_block *blocks, uint32_t block_count,
        const void *render, uint64_t render_length,
        const int32_t *tokens, uint32_t token_count,
        const void *alternate_render, uint64_t alternate_render_length,
        const int32_t *alternate_tokens, uint32_t alternate_token_count,
        const void *reasoning, uint64_t reasoning_length,
        uint32_t reasoning_close) {
    if (!c || (block_count && !blocks) || (render_length && !render) ||
        (token_count && !tokens) ||
        (alternate_render_length && !alternate_render) ||
        (alternate_token_count && !alternate_tokens) ||
        (reasoning_length && !reasoning) ||
        reasoning_close > CONVERSATION_REASONING_EOS)
        return CONVERSATION_INVALID_ARGUMENT;
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    int ok = conversation_buffer_field_u32(&payload, 1, role) &&
        conversation_buffer_blocks(&payload, 2, blocks, block_count) &&
        conversation_buffer_field(&payload, 3, render, render_length) &&
        conversation_buffer_tokens(&payload, 4, tokens, token_count);
    if (ok && (alternate_render || alternate_tokens))
        ok = conversation_buffer_field(
                 &payload, CONVERSATION_FIELD_OPTIONAL | 5,
                 alternate_render, alternate_render_length) &&
             conversation_buffer_tokens(
                 &payload, CONVERSATION_FIELD_OPTIONAL | 6,
                 alternate_tokens, alternate_token_count);
    if (ok && reasoning)
        ok = conversation_buffer_field(
            &payload, CONVERSATION_FIELD_OPTIONAL | 7,
            reasoning, reasoning_length);
    if (ok && reasoning_close != CONVERSATION_REASONING_NONE)
        ok = conversation_buffer_field_u32(
            &payload, CONVERSATION_FIELD_OPTIONAL | 8, reasoning_close);
    if (ok)
        status = conversation_frame_write(c, CONVERSATION_EVENT_MESSAGE,
                                          CONVERSATION_FRAME_CRITICAL,
                                          &payload);
    free(payload.data);
    return status;
}

conversation_status conversation_append_generation_started(
        conversation *c, const conversation_generation *generation) {
    if (!c || !generation ||
        generation->settings.reasoning_effort >
            CONVERSATION_REASONING_MAX ||
        generation->settings.reasoning_history >
            CONVERSATION_REASONING_PRESERVE_TOOL_CALLS ||
        generation->settings.reasoning_budget < -1)
        return CONVERSATION_INVALID_ARGUMENT;
    uint8_t packed[CONVERSATION_SETTINGS_SIZE];
    conversation_settings_pack(&generation->settings, packed);
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    if (conversation_buffer_field_u64(&payload, 1,
                                      generation->generation_id) &&
        conversation_buffer_field(&payload, 2, packed, sizeof packed))
        status = conversation_frame_write(
            c, CONVERSATION_EVENT_GENERATION_STARTED,
            CONVERSATION_FRAME_CRITICAL, &payload);
    free(payload.data);
    return status;
}

conversation_status conversation_append_generation_result(
        conversation *c, uint64_t generation_id, uint32_t stop_reason,
        uint64_t rng_after,
        const conversation_block *blocks, uint32_t block_count,
        const void *render, uint64_t render_length,
        const int32_t *tokens, uint32_t token_count) {
    return conversation_append_generation_result_variants(
        c, generation_id, stop_reason, rng_after, blocks, block_count,
        render, render_length, tokens, token_count, NULL, 0, NULL, 0,
        NULL, 0, CONVERSATION_REASONING_NONE, NULL, 0, NULL, 0);
}

conversation_status conversation_append_generation_result_variants(
        conversation *c, uint64_t generation_id, uint32_t stop_reason,
        uint64_t rng_after,
        const conversation_block *blocks, uint32_t block_count,
        const void *render, uint64_t render_length,
        const int32_t *tokens, uint32_t token_count,
        const void *alternate_render, uint64_t alternate_render_length,
        const int32_t *alternate_tokens, uint32_t alternate_token_count,
        const void *reasoning, uint64_t reasoning_length,
        uint32_t reasoning_close,
        const void *raw_render, uint64_t raw_render_length,
        const int32_t *raw_tokens, uint32_t raw_token_count) {
    if (!c || (block_count && !blocks) || (render_length && !render) ||
        (token_count && !tokens) ||
        (alternate_render_length && !alternate_render) ||
        (alternate_token_count && !alternate_tokens) ||
        (reasoning_length && !reasoning) ||
        (raw_render_length && !raw_render) ||
        (raw_token_count && !raw_tokens) ||
        reasoning_close > CONVERSATION_REASONING_EOS)
        return CONVERSATION_INVALID_ARGUMENT;
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    int ok = conversation_buffer_field_u64(&payload, 1, generation_id) &&
        conversation_buffer_field_u32(&payload, 2, stop_reason) &&
        conversation_buffer_field_u64(&payload, 3, rng_after) &&
        conversation_buffer_blocks(&payload, 4, blocks, block_count) &&
        conversation_buffer_field(&payload, 5, render, render_length) &&
        conversation_buffer_tokens(&payload, 6, tokens, token_count);
    if (ok && (alternate_render || alternate_tokens))
        ok = conversation_buffer_field(
                 &payload, CONVERSATION_FIELD_OPTIONAL | 7,
                 alternate_render, alternate_render_length) &&
             conversation_buffer_tokens(
                 &payload, CONVERSATION_FIELD_OPTIONAL | 8,
                 alternate_tokens, alternate_token_count);
    if (ok && reasoning)
        ok = conversation_buffer_field(
            &payload, CONVERSATION_FIELD_OPTIONAL | 9,
            reasoning, reasoning_length);
    if (ok && reasoning_close != CONVERSATION_REASONING_NONE)
        ok = conversation_buffer_field_u32(
            &payload, CONVERSATION_FIELD_OPTIONAL | 10, reasoning_close);
    if (ok && (raw_render || raw_tokens))
        ok = conversation_buffer_field(
                 &payload, CONVERSATION_FIELD_OPTIONAL | 11,
                 raw_render, raw_render_length) &&
             conversation_buffer_tokens(
                 &payload, CONVERSATION_FIELD_OPTIONAL | 12,
                 raw_tokens, raw_token_count);
    if (ok)
        status = conversation_frame_write(
            c, CONVERSATION_EVENT_GENERATION_RESULT,
            CONVERSATION_FRAME_CRITICAL, &payload);
    free(payload.data);
    return status;
}

conversation_status conversation_append_tool_started(
        conversation *c, const conversation_tool_call *call) {
    if (!c || !call || !call->server || !call->tool ||
        (call->arguments_length && !call->arguments))
        return CONVERSATION_INVALID_ARGUMENT;
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    if (conversation_buffer_field_u64(&payload, 1, call->call_id) &&
        conversation_buffer_field(&payload, 2, call->server,
                                  strlen(call->server)) &&
        conversation_buffer_field(&payload, 3, call->tool,
                                  strlen(call->tool)) &&
        conversation_buffer_field(&payload, 4, call->arguments,
                                  call->arguments_length) &&
        conversation_buffer_field(&payload, 5, call->fingerprint, 32))
        status = conversation_frame_write(c, CONVERSATION_EVENT_TOOL_STARTED,
                                          CONVERSATION_FRAME_CRITICAL,
                                          &payload);
    free(payload.data);
    return status;
}

conversation_status conversation_append_tool_result(
        conversation *c, uint64_t call_id, uint32_t tool_status,
        const conversation_block *blocks, uint32_t block_count,
        const void *render, uint64_t render_length,
        const int32_t *tokens, uint32_t token_count) {
    if (!c || (block_count && !blocks) || (render_length && !render) ||
        (token_count && !tokens)) return CONVERSATION_INVALID_ARGUMENT;
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    if (conversation_buffer_field_u64(&payload, 1, call_id) &&
        conversation_buffer_field_u32(&payload, 2, tool_status) &&
        conversation_buffer_blocks(&payload, 3, blocks, block_count) &&
        conversation_buffer_field(&payload, 4, render, render_length) &&
        conversation_buffer_tokens(&payload, 5, tokens, token_count))
        status = conversation_frame_write(c, CONVERSATION_EVENT_TOOL_RESULT,
                                          CONVERSATION_FRAME_CRITICAL,
                                          &payload);
    free(payload.data);
    return status;
}

conversation_status conversation_append_snapshot_ref(
        conversation *c, const kvstore_id *snapshot, uint64_t token_boundary) {
    if (!c || !snapshot) return CONVERSATION_INVALID_ARGUMENT;
    if (token_boundary > c->token_count) return CONVERSATION_FORMAT;
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    if (conversation_buffer_field(&payload, 1, snapshot->bytes, 16) &&
        conversation_buffer_field_u64(&payload, 2, token_boundary) &&
        conversation_buffer_field_u64(&payload, 3, c->epoch) &&
        conversation_buffer_field_u64(&payload, 4, c->event_count))
        status = conversation_frame_write(c, CONVERSATION_EVENT_SNAPSHOT_REF,
                                          CONVERSATION_FRAME_CRITICAL,
                                          &payload);
    free(payload.data);
    return status;
}

conversation_status conversation_append_cache_epoch(conversation *c) {
    if (!c) return CONVERSATION_INVALID_ARGUMENT;
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    if (conversation_buffer_field_u64(&payload, 1, c->epoch + 1))
        status = conversation_frame_write(c, CONVERSATION_EVENT_CACHE_EPOCH,
                                          CONVERSATION_FRAME_CRITICAL,
                                          &payload);
    free(payload.data);
    return status;
}

conversation_status conversation_append_rewind(conversation *c,
                                               uint64_t target_event) {
    if (!c) return CONVERSATION_INVALID_ARGUMENT;
    conversation_buffer payload = {0};
    conversation_status status = CONVERSATION_NOMEM;
    if (conversation_buffer_field_u64(&payload, 1, target_event))
        status = conversation_frame_write(c, CONVERSATION_EVENT_REWIND,
                                          CONVERSATION_FRAME_CRITICAL,
                                          &payload);
    free(payload.data);
    return status;
}

const conversation_id *conversation_get_id(const conversation *c) {
    return &c->id;
}

int64_t conversation_created(const conversation *c) {
    return c->created;
}

int64_t conversation_updated(const conversation *c) {
    return c->updated;
}

const char *conversation_get_title(const conversation *c) {
    return c->title;
}

const char *conversation_get_workspace(const conversation *c) {
    return c->workspace;
}

int conversation_get_settings(const conversation *c,
                              conversation_settings *out) {
    if (!c->has_settings) return 0;
    *out = c->settings;
    return 1;
}

uint64_t conversation_epoch_current(const conversation *c) {
    return c->epoch;
}

const int32_t *conversation_tokens(const conversation *c, uint64_t *count) {
    if (count) *count = c->token_count;
    return c->tokens;
}

static int conversation_event_has_calls(const conversation_event *event) {
    return event->role == CONVERSATION_ROLE_ASSISTANT &&
           event->block_count > 1 &&
           event->blocks[1].format == CONVERSATION_BLOCK_JSON;
}

static int conversation_event_variant(const conversation_event *event,
                                      int thinking,
                                      uint32_t reasoning_history,
                                      uint64_t position,
                                      uint64_t last_user,
                                      int current_open) {
    int variant = 0;
    if (!thinking && event->raw_tokens && !event->alternate_tokens)
        variant = 2;
    if (thinking && event->role == CONVERSATION_ROLE_SYSTEM)
        variant = event->alternate_tokens != NULL;
    if (thinking && event->role == CONVERSATION_ROLE_ASSISTANT) {
        int current = current_open &&
                      (last_user == UINT64_MAX || position > last_user);
        int historical =
            reasoning_history ==
                CONVERSATION_REASONING_PRESERVE_TOOL_CALLS &&
            conversation_event_has_calls(event);
        variant = (current || historical) &&
                  event->alternate_tokens != NULL;
    }
    return variant;
}

conversation_status conversation_project_copy(const conversation *c,
                                               int thinking,
                                               uint32_t reasoning_history,
                                               int close_current,
                                               int32_t **tokens,
                                               uint64_t *count) {
    if (!c || !tokens || !count || reasoning_history >
              CONVERSATION_REASONING_PRESERVE_TOOL_CALLS)
        return CONVERSATION_INVALID_ARGUMENT;
    *tokens = NULL;
    *count = 0;
    uint64_t last_user = UINT64_MAX;
    for (uint64_t i = 0; i < c->visible_count; i++) {
        const conversation_event *event = &c->events[c->visible[i].index].view;
        if (event->role == CONVERSATION_ROLE_USER) last_user = i;
    }
    int current_open = 0;
    if (!close_current && c->visible_count) {
        const conversation_event *last =
            &c->events[c->visible[c->visible_count - 1].index].view;
        if (last->type == CONVERSATION_EVENT_TOOL_RESULT ||
            conversation_event_has_calls(last) ||
            (last->type == CONVERSATION_EVENT_GENERATION_RESULT &&
             (last->stop_reason == CONVERSATION_STOP_LIMIT ||
              last->stop_reason == CONVERSATION_STOP_CANCELLED)))
            current_open = 1;
    }
    uint64_t total = 0;
    for (uint64_t i = 0; i < c->visible_count; i++) {
        const conversation_event *event = &c->events[c->visible[i].index].view;
        int variant = conversation_event_variant(
            event, thinking, reasoning_history, i, last_user, current_open);
        uint32_t n = variant == 1 ? event->alternate_token_count
                     : variant == 2 ? event->raw_token_count
                                    : event->token_count;
        if (total > CONVERSATION_TOKEN_LIMIT - n)
            return CONVERSATION_LIMIT;
        total += n;
    }
    int32_t *projected = malloc((size_t)(total ? total : 1) *
                                sizeof(*projected));
    if (!projected) return CONVERSATION_NOMEM;
    uint64_t offset = 0;
    for (uint64_t i = 0; i < c->visible_count; i++) {
        const conversation_event *event = &c->events[c->visible[i].index].view;
        int variant = conversation_event_variant(
            event, thinking, reasoning_history, i, last_user, current_open);
        const int32_t *source = variant == 1 ? event->alternate_tokens
                                : variant == 2 ? event->raw_tokens
                                               : event->tokens;
        uint32_t n = variant == 1 ? event->alternate_token_count
                     : variant == 2 ? event->raw_token_count
                                    : event->token_count;
        if (n) memcpy(projected + offset, source,
                      (size_t)n * sizeof(*source));
        offset += n;
    }
    *tokens = projected;
    *count = total;
    return CONVERSATION_OK;
}

conversation_status conversation_project(conversation *c, int thinking,
                                         uint32_t reasoning_history) {
    if (!c || reasoning_history >
              CONVERSATION_REASONING_PRESERVE_TOOL_CALLS)
        return CONVERSATION_INVALID_ARGUMENT;
    uint64_t last_user = UINT64_MAX;
    for (uint64_t i = 0; i < c->visible_count; i++) {
        const conversation_event *event = &c->events[c->visible[i].index].view;
        if (event->role == CONVERSATION_ROLE_USER) last_user = i;
    }
    int current_open = 0;
    if (c->visible_count) {
        const conversation_event *last =
            &c->events[c->visible[c->visible_count - 1].index].view;
        if (last->type == CONVERSATION_EVENT_TOOL_RESULT ||
            conversation_event_has_calls(last) ||
            (last->type == CONVERSATION_EVENT_GENERATION_RESULT &&
             (last->stop_reason == CONVERSATION_STOP_LIMIT ||
              last->stop_reason == CONVERSATION_STOP_CANCELLED)))
            current_open = 1;
    }
    uint64_t total = 0;
    for (uint64_t i = 0; i < c->visible_count; i++) {
        const conversation_event *event = &c->events[c->visible[i].index].view;
        int variant = conversation_event_variant(
            event, thinking, reasoning_history, i, last_user, current_open);
        uint32_t count = variant == 1 ? event->alternate_token_count
                         : variant == 2 ? event->raw_token_count
                                        : event->token_count;
        if (total > CONVERSATION_TOKEN_LIMIT - count)
            return CONVERSATION_LIMIT;
        total += count;
    }
    int32_t *projected = malloc((size_t)(total ? total : 1) *
                                sizeof(*projected));
    if (!projected) return CONVERSATION_NOMEM;
    uint64_t offset = 0;
    for (uint64_t i = 0; i < c->visible_count; i++) {
        const conversation_event *event = &c->events[c->visible[i].index].view;
        int variant = conversation_event_variant(
            event, thinking, reasoning_history, i, last_user, current_open);
        const int32_t *tokens = variant == 1 ? event->alternate_tokens
                                : variant == 2 ? event->raw_tokens
                                               : event->tokens;
        uint32_t count = variant == 1 ? event->alternate_token_count
                         : variant == 2 ? event->raw_token_count
                                        : event->token_count;
        if (count) memcpy(projected + offset, tokens,
                          (size_t)count * sizeof(*tokens));
        offset += count;
        c->visible[i].boundary = offset;
    }
    int snapshot_changed = c->has_snapshot &&
        (c->snapshot_boundary > total ||
         (c->snapshot_boundary &&
          memcmp(c->tokens, projected,
                 (size_t)c->snapshot_boundary * sizeof(*projected)) != 0));
    free(c->tokens);
    c->tokens = projected;
    c->token_count = total;
    c->token_capacity = total;
    if (snapshot_changed) {
        c->has_snapshot = 0;
        c->snapshot_boundary = 0;
    }
    return CONVERSATION_OK;
}

int conversation_snapshot_current(const conversation *c, kvstore_id *id,
                                  uint64_t *boundary) {
    if (!c->has_snapshot) return 0;
    if (id) *id = c->snapshot;
    if (boundary) *boundary = c->snapshot_boundary;
    return 1;
}

uint64_t conversation_event_count(const conversation *c) {
    return c->event_count;
}

const conversation_event *conversation_event_at(const conversation *c,
                                                uint64_t index) {
    if (index >= c->event_count) return NULL;
    return &c->events[index].view;
}

uint64_t conversation_visible_count(const conversation *c) {
    return c->visible_count;
}

uint64_t conversation_visible_index(const conversation *c,
                                    uint64_t position) {
    if (position >= c->visible_count) return UINT64_MAX;
    return c->visible[position].index;
}

uint64_t conversation_visible_boundary(const conversation *c,
                                       uint64_t position) {
    if (position >= c->visible_count) return 0;
    return c->visible[position].boundary;
}

int conversation_visible_position(const conversation *c,
                                  uint64_t event_index, uint64_t *position) {
    for (uint64_t i = 0; i < c->visible_count; i++)
        if (c->visible[i].index == event_index) {
            if (position) *position = i;
            return 1;
        }
    return 0;
}

int conversation_generation_interrupted(const conversation *c) {
    return c->generation_pending;
}

size_t conversation_unknown_tool_calls(const conversation *c,
                                       uint64_t *call_ids, size_t cap) {
    size_t unknown = 0;
    for (size_t i = 0; i < c->call_count; i++) {
        if (c->calls[i].has_result) continue;
        if (call_ids && unknown < cap) call_ids[unknown] = c->calls[i].call_id;
        unknown++;
    }
    return unknown;
}

uint64_t conversation_uncommitted(const conversation *c) {
    return c->uncommitted;
}

conversation_status conversation_resume(conversation *c, kvstore *store,
                                        xe_session *session,
                                        conversation_resume_report *report) {
    if (!c || !session) return CONVERSATION_INVALID_ARGUMENT;
    if (report) memset(report, 0, sizeof *report);
    if (c->token_count > (uint64_t)xe_session_context_size(session))
        return CONVERSATION_LIMIT;
    if (c->token_count == 0) {
        xe_session_reset(session);
        return CONVERSATION_OK;
    }
    xe_tokens full = { c->tokens, (int)c->token_count,
                       (int)c->token_count };
    if (store && c->has_snapshot && c->snapshot_boundary > 0 &&
        c->snapshot_boundary <= c->token_count) {
        xe_tokens prefix = { c->tokens, (int)c->snapshot_boundary,
                             (int)c->snapshot_boundary };
        xe_snapshot_status core;
        kvstore_status loaded = kvstore_load(store, &c->snapshot, session,
                                             &prefix, &core);
        if (loaded == KVSTORE_OK && report) {
            report->used_snapshot = 1;
            report->boundary = c->snapshot_boundary;
            report->zero_prefill = c->snapshot_boundary == c->token_count;
        }
    }
    xe_session_sync(session, &full);
    return CONVERSATION_OK;
}

int conversation_autosave_due(uint64_t tokens, uint64_t saved_tokens,
                              int64_t saved_at, int64_t now) {
    if (tokens <= saved_tokens) return 0;
    uint64_t need = saved_tokens / 4;
    if (need < 512) need = 512;
    if (tokens - saved_tokens < need) return 0;
    if (now < saved_at || now - saved_at < INT64_C(30000000000)) return 0;
    return 1;
}

const char *conversation_status_name(conversation_status status) {
    static const char *names[] = {
        "ok", "miss", "invalid-argument", "io", "format", "version",
        "damaged", "locked", "limit", "no-memory"
    };
    return (unsigned)status < sizeof names / sizeof names[0]
           ? names[status] : "unknown";
}
