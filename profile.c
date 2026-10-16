#include "profile.h"
#include "format.h"
#include "json.h"

#include <stdlib.h>
#include <string.h>

enum {
    PROFILE_ID_EOS = 1,
    PROFILE_ID_BOS = 2,
    PROFILE_ID_TOOL_BEGIN = 46,
    PROFILE_ID_TOOL_END = 47,
    PROFILE_ID_CALL_BEGIN = 48,
    PROFILE_ID_CALL_END = 49,
    PROFILE_ID_RESPONSE_BEGIN = 50,
    PROFILE_ID_RESPONSE_END = 51,
    PROFILE_ID_QUOTE = 52,
    PROFILE_ID_THINK = 98,
    PROFILE_ID_CHANNEL_BEGIN = 100,
    PROFILE_ID_CHANNEL_END = 101,
    PROFILE_ID_TURN_BEGIN = 105,
    PROFILE_ID_TURN_END = 106,
    PROFILE_ID_IMAGE_BEGIN = 255999,
    PROFILE_ID_IMAGE = 258880,
    PROFILE_ID_IMAGE_END = 258882,
    PROFILE_MAX_DEPTH = 64,
    PROFILE_DETOK_MAX = 512
};

enum {
    PROFILE_PS_CONTENT,
    PROFILE_PS_CHANNEL,
    PROFILE_PS_THOUGHT,
    PROFILE_PS_NAME,
    PROFILE_PS_ARGS,
    PROFILE_PS_CALL_CLOSE,
    PROFILE_PS_DONE
};

enum {
    PROFILE_ARG_KEY,
    PROFILE_ARG_VALUE,
    PROFILE_ARG_BARE,
    PROFILE_ARG_AFTER
};

static const uint8_t profile_template_sha[32] = {
    0x84, 0x5f, 0x1e, 0xe4, 0x8e, 0x39, 0xfc, 0x94,
    0x2f, 0xe1, 0x90, 0xda, 0x9d, 0xf6, 0xa1, 0xc5,
    0xdb, 0x22, 0x9e, 0x17, 0xa9, 0x6e, 0xa0, 0x89,
    0x66, 0xad, 0x1c, 0x92, 0x74, 0xe7, 0x3d, 0x1b
};

struct profile {
    const xe_engine *engine;
    uint8_t *render;
    size_t render_length;
    size_t render_capacity;
    int32_t *tokens;
    uint32_t token_count;
    uint32_t token_capacity;
    char *run;
    size_t run_length;
    size_t run_capacity;
    int failed;

    int state;
    int in_string;
    size_t calls_done;
    char *scan;
    size_t scan_length;
    size_t scan_capacity;
    char *name;
    size_t name_length;
    size_t name_capacity;
    json_writer args;
    uint8_t level_kind[PROFILE_MAX_DEPTH];
    uint8_t level_phase[PROFILE_MAX_DEPTH];
    int depth;
    char detok[PROFILE_DETOK_MAX];
};

static const char *profile_piece(int32_t id) {
    switch (id) {
    case PROFILE_ID_EOS: return "<eos>";
    case PROFILE_ID_BOS: return "<bos>";
    case PROFILE_ID_TOOL_BEGIN: return "<|tool>";
    case PROFILE_ID_TOOL_END: return "<tool|>";
    case PROFILE_ID_CALL_BEGIN: return "<|tool_call>";
    case PROFILE_ID_CALL_END: return "<tool_call|>";
    case PROFILE_ID_RESPONSE_BEGIN: return "<|tool_response>";
    case PROFILE_ID_RESPONSE_END: return "<tool_response|>";
    case PROFILE_ID_QUOTE: return "<|\"|>";
    case PROFILE_ID_THINK: return "<|think|>";
    case PROFILE_ID_CHANNEL_BEGIN: return "<|channel>";
    case PROFILE_ID_CHANNEL_END: return "<channel|>";
    case PROFILE_ID_TURN_BEGIN: return "<|turn>";
    case PROFILE_ID_IMAGE_BEGIN: return "<|image>";
    case PROFILE_ID_IMAGE: return "<|image|>";
    case PROFILE_ID_IMAGE_END: return "<image|>";
    case PROFILE_ID_TURN_END: return "<turn|>";
    default: return "";
    }
}

const char *profile_model(const profile *p) {
    (void)p;
    return "gemma-4-26B-A4B-it-qat";
}

profile_status profile_open(profile **out, const xe_engine *engine) {
    if (!out || !engine) return PROFILE_INVALID_ARGUMENT;
    *out = NULL;
    static const struct {
        const char *piece;
        int32_t id;
    } markers[] = {
        { "<|tool>", PROFILE_ID_TOOL_BEGIN },
        { "<tool|>", PROFILE_ID_TOOL_END },
        { "<|tool_call>", PROFILE_ID_CALL_BEGIN },
        { "<tool_call|>", PROFILE_ID_CALL_END },
        { "<|tool_response>", PROFILE_ID_RESPONSE_BEGIN },
        { "<tool_response|>", PROFILE_ID_RESPONSE_END },
        { "<|\"|>", PROFILE_ID_QUOTE },
        { "<|think|>", PROFILE_ID_THINK },
        { "<|channel>", PROFILE_ID_CHANNEL_BEGIN },
        { "<channel|>", PROFILE_ID_CHANNEL_END },
        { "<|turn>", PROFILE_ID_TURN_BEGIN },
        { "<turn|>", PROFILE_ID_TURN_END },
    };
    for (size_t i = 0; i < sizeof markers / sizeof markers[0]; i++)
        if (xe_token_id(engine, markers[i].piece) != markers[i].id)
            return PROFILE_TEMPLATE_MISMATCH;
    if (xe_bos_id(engine) != PROFILE_ID_BOS ||
        xe_eos_id(engine) != PROFILE_ID_EOS ||
        xe_eot_id(engine) != PROFILE_ID_TURN_END)
        return PROFILE_TEMPLATE_MISMATCH;
    uint64_t template_length = 0;
    const char *template_bytes = xe_chat_template(engine, &template_length);
    if (!template_bytes || !template_length)
        return PROFILE_TEMPLATE_MISMATCH;
    uint8_t digest[32];
    format_sha256_bytes(template_bytes, (size_t)template_length, digest);
    if (memcmp(digest, profile_template_sha, 32) != 0)
        return PROFILE_TEMPLATE_MISMATCH;
    profile *p = calloc(1, sizeof *p);
    if (!p) return PROFILE_NOMEM;
    p->engine = engine;
    p->state = PROFILE_PS_CONTENT;
    *out = p;
    return PROFILE_OK;
}

void profile_close(profile *p) {
    if (!p) return;
    free(p->render);
    free(p->tokens);
    free(p->run);
    free(p->scan);
    free(p->name);
    json_writer_free(&p->args);
    free(p);
}

static int profile_grow(void **data, size_t *capacity, size_t need,
                        size_t item) {
    if (need <= *capacity) return 1;
    size_t capacity_next = *capacity ? *capacity : 256;
    while (capacity_next < need) {
        if (capacity_next > SIZE_MAX / 2) return 0;
        capacity_next *= 2;
    }
    void *grown = realloc(*data, capacity_next * item);
    if (!grown) return 0;
    *data = grown;
    *capacity = capacity_next;
    return 1;
}

static int prof_render_put(profile *p, const void *data, size_t length) {
    if (p->failed) return 0;
    void *buffer = p->render;
    if (!profile_grow(&buffer, &p->render_capacity,
                      p->render_length + length, 1)) {
        p->failed = 1;
        return 0;
    }
    p->render = buffer;
    memcpy(p->render + p->render_length, data, length);
    p->render_length += length;
    return 1;
}

static int prof_run_put(profile *p, const char *text, size_t length) {
    if (p->failed) return 0;
    void *buffer = p->run;
    if (!profile_grow(&buffer, &p->run_capacity,
                      p->run_length + length + 1, 1)) {
        p->failed = 1;
        return 0;
    }
    p->run = buffer;
    memcpy(p->run + p->run_length, text, length);
    p->run_length += length;
    p->run[p->run_length] = '\0';
    return 1;
}

static int prof_run_text(profile *p, const char *text) {
    return prof_run_put(p, text, strlen(text));
}

static int prof_token_put(profile *p, int32_t token) {
    if (p->failed) return 0;
    void *buffer = p->tokens;
    size_t capacity = p->token_capacity;
    if (!profile_grow(&buffer, &capacity, (size_t)p->token_count + 1,
                      sizeof(int32_t))) {
        p->failed = 1;
        return 0;
    }
    p->tokens = buffer;
    p->token_capacity = (uint32_t)capacity;
    p->tokens[p->token_count++] = token;
    return 1;
}

static int prof_flush(profile *p) {
    if (p->failed) return 0;
    if (!p->run_length) return 1;
    if (p->run_length > (SIZE_MAX - 16) / 3 ||
        3 * p->run_length + 16 > UINT32_MAX) {
        p->failed = 1;
        return 0;
    }
    uint32_t extra = (uint32_t)(3 * p->run_length + 16);
    void *buffer = p->tokens;
    size_t capacity = p->token_capacity;
    if (!profile_grow(&buffer, &capacity,
                      (size_t)p->token_count + extra, sizeof(int32_t))) {
        p->failed = 1;
        return 0;
    }
    p->tokens = buffer;
    p->token_capacity = (uint32_t)capacity;
    int encoded = xe_encode_text(p->engine, p->run,
                                 p->tokens + p->token_count, (int)extra);
    if (encoded < 0) {
        p->failed = 1;
        return 0;
    }
    p->token_count += (uint32_t)encoded;
    if (!prof_render_put(p, p->run, p->run_length)) return 0;
    p->run_length = 0;
    p->run[0] = '\0';
    return 1;
}

static int prof_marker(profile *p, int32_t id) {
    if (!prof_flush(p)) return 0;
    const char *piece = profile_piece(id);
    return prof_token_put(p, id) &&
           prof_render_put(p, piece, strlen(piece));
}

static void prof_begin(profile *p) {
    p->render_length = 0;
    p->token_count = 0;
    p->run_length = 0;
    if (p->run) p->run[0] = '\0';
    p->failed = 0;
}

static profile_status prof_finish(profile *p, profile_render *out) {
    if (!prof_flush(p)) return PROFILE_NOMEM;
    out->render = p->render;
    out->render_length = p->render_length;
    out->tokens = p->tokens;
    out->token_count = p->token_count;
    return PROFILE_OK;
}

static void prof_trim(const char *text, const char **begin, size_t *length) {
    const char *start = text;
    const char *end = text + strlen(text);
    while (start < end &&
           (*start == ' ' || *start == '\t' || *start == '\n' ||
            *start == '\r' || *start == '\v' || *start == '\f'))
        start++;
    while (end > start &&
           (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n' ||
            end[-1] == '\r' || end[-1] == '\v' || end[-1] == '\f'))
        end--;
    *begin = start;
    *length = (size_t)(end - start);
}

static int prof_quoted(profile *p, const char *text, size_t length) {
    return prof_marker(p, PROFILE_ID_QUOTE) &&
           prof_run_put(p, text, length) &&
           prof_marker(p, PROFILE_ID_QUOTE);
}

static int prof_compare_keys(const void *a, const void *b) {
    const json_value *const *va = a;
    const json_value *const *vb = b;
    return strcmp((*va)->key, (*vb)->key);
}

static const json_value **prof_sorted(const json_value *object,
                                      size_t *count) {
    size_t n = json_length(object);
    *count = n;
    const json_value **items = malloc(n ? n * sizeof *items : 1);
    if (!items) return NULL;
    size_t i = 0;
    for (const json_value *v = object->child; v; v = v->next)
        items[i++] = v;
    qsort(items, n, sizeof *items, prof_compare_keys);
    return items;
}

static profile_status prof_dsl_value(profile *p, const json_value *value,
                                     int escape_keys) {
    switch (value->type) {
    case JSON_NULL:
        return prof_run_text(p, "null") ? PROFILE_OK : PROFILE_NOMEM;
    case JSON_BOOL:
        return prof_run_text(p, value->boolean ? "true" : "false")
               ? PROFILE_OK : PROFILE_NOMEM;
    case JSON_NUMBER:
        return prof_run_put(p, value->text, value->text_length)
               ? PROFILE_OK : PROFILE_NOMEM;
    case JSON_STRING:
        return prof_quoted(p, value->text, value->text_length)
               ? PROFILE_OK : PROFILE_NOMEM;
    case JSON_ARRAY: {
        if (!prof_run_text(p, "[")) return PROFILE_NOMEM;
        int first = 1;
        for (const json_value *v = value->child; v; v = v->next) {
            if (!first && !prof_run_text(p, ",")) return PROFILE_NOMEM;
            first = 0;
            profile_status status = prof_dsl_value(p, v, escape_keys);
            if (status != PROFILE_OK) return status;
        }
        return prof_run_text(p, "]") ? PROFILE_OK : PROFILE_NOMEM;
    }
    case JSON_OBJECT: {
        if (!prof_run_text(p, "{")) return PROFILE_NOMEM;
        size_t count;
        const json_value **items = prof_sorted(value, &count);
        if (!items) return PROFILE_NOMEM;
        profile_status status = PROFILE_OK;
        for (size_t i = 0; i < count && status == PROFILE_OK; i++) {
            if (i && !prof_run_text(p, ",")) {
                status = PROFILE_NOMEM;
                break;
            }
            if (escape_keys) {
                if (!prof_quoted(p, items[i]->key, strlen(items[i]->key)))
                    status = PROFILE_NOMEM;
            } else {
                if (!prof_run_text(p, items[i]->key))
                    status = PROFILE_NOMEM;
            }
            if (status == PROFILE_OK && !prof_run_text(p, ":"))
                status = PROFILE_NOMEM;
            if (status == PROFILE_OK)
                status = prof_dsl_value(p, items[i], escape_keys);
        }
        free(items);
        if (status != PROFILE_OK) return status;
        return prof_run_text(p, "}") ? PROFILE_OK : PROFILE_NOMEM;
    }
    default:
        return PROFILE_INVALID_ARGUMENT;
    }
}

static int prof_type_upper(const json_value *type, char out[24]) {
    if (!type || type->type != JSON_STRING || type->text_length >= 24)
        return 0;
    for (size_t i = 0; i < type->text_length; i++) {
        char c = type->text[i];
        out[i] = c >= 'a' && c <= 'z' ? (char)(c - 32) : c;
    }
    out[type->text_length] = '\0';
    return 1;
}

static profile_status prof_required_list(profile *p,
                                         const json_value *required) {
    if (!prof_run_text(p, "[")) return PROFILE_NOMEM;
    int first = 1;
    for (const json_value *v = required->child; v; v = v->next) {
        if (v->type != JSON_STRING) return PROFILE_INVALID_ARGUMENT;
        if (!first && !prof_run_text(p, ",")) return PROFILE_NOMEM;
        first = 0;
        if (!prof_quoted(p, v->text, v->text_length)) return PROFILE_NOMEM;
    }
    return prof_run_text(p, "]") ? PROFILE_OK : PROFILE_NOMEM;
}

static profile_status prof_format_properties(profile *p,
                                             const json_value *properties);

static profile_status prof_format_items(profile *p, const json_value *items) {
    if (!prof_run_text(p, "items:{")) return PROFILE_NOMEM;
    size_t count;
    const json_value **sorted = prof_sorted(items, &count);
    if (!sorted) return PROFILE_NOMEM;
    profile_status status = PROFILE_OK;
    int first = 1;
    for (size_t i = 0; i < count && status == PROFILE_OK; i++) {
        const json_value *v = sorted[i];
        if (v->type == JSON_NULL) continue;
        if (!first && !prof_run_text(p, ",")) {
            status = PROFILE_NOMEM;
            break;
        }
        first = 0;
        if (strcmp(v->key, "properties") == 0 && v->type == JSON_OBJECT) {
            if (!prof_run_text(p, "properties:{"))
                status = PROFILE_NOMEM;
            if (status == PROFILE_OK)
                status = prof_format_properties(p, v);
            if (status == PROFILE_OK && !prof_run_text(p, "}"))
                status = PROFILE_NOMEM;
        } else if (strcmp(v->key, "required") == 0 &&
                   v->type == JSON_ARRAY) {
            if (!prof_run_text(p, "required:"))
                status = PROFILE_NOMEM;
            if (status == PROFILE_OK)
                status = prof_required_list(p, v);
        } else if (strcmp(v->key, "type") == 0) {
            if (v->type == JSON_STRING) {
                char upper[24];
                if (!prof_type_upper(v, upper))
                    status = PROFILE_INVALID_ARGUMENT;
                else if (!prof_run_text(p, "type:") ||
                         !prof_quoted(p, upper, strlen(upper)))
                    status = PROFILE_NOMEM;
            } else if (v->type == JSON_ARRAY) {
                if (!prof_run_text(p, "type:["))
                    status = PROFILE_NOMEM;
                int type_first = 1;
                for (const json_value *t = v->child;
                     t && status == PROFILE_OK; t = t->next) {
                    char upper[24];
                    if (!prof_type_upper(t, upper)) {
                        status = PROFILE_INVALID_ARGUMENT;
                        break;
                    }
                    if (!type_first && !prof_run_text(p, ",")) {
                        status = PROFILE_NOMEM;
                        break;
                    }
                    type_first = 0;
                    if (!prof_quoted(p, upper, strlen(upper)))
                        status = PROFILE_NOMEM;
                }
                if (status == PROFILE_OK && !prof_run_text(p, "]"))
                    status = PROFILE_NOMEM;
            } else {
                status = PROFILE_INVALID_ARGUMENT;
            }
        } else {
            if (!prof_run_text(p, v->key) || !prof_run_text(p, ":"))
                status = PROFILE_NOMEM;
            if (status == PROFILE_OK)
                status = prof_dsl_value(p, v, 1);
        }
    }
    free(sorted);
    if (status != PROFILE_OK) return status;
    return prof_run_text(p, "}") ? PROFILE_OK : PROFILE_NOMEM;
}

static profile_status prof_format_property(profile *p, const char *key,
                                           const json_value *value) {
    if (value->type != JSON_OBJECT) return PROFILE_INVALID_ARGUMENT;
    char upper[24];
    if (!prof_type_upper(json_member(value, "type"), upper))
        return PROFILE_INVALID_ARGUMENT;
    if (!prof_run_text(p, key) || !prof_run_text(p, ":{"))
        return PROFILE_NOMEM;
    int add_comma = 0;
    const json_value *description = json_member(value, "description");
    if (description && description->type == JSON_STRING) {
        if (!prof_run_text(p, "description:") ||
            !prof_quoted(p, description->text, description->text_length))
            return PROFILE_NOMEM;
        add_comma = 1;
    }
    const json_value *member;
    if (strcmp(upper, "STRING") == 0 &&
        (member = json_member(value, "enum")) != NULL &&
        member->type == JSON_ARRAY) {
        if (add_comma && !prof_run_text(p, ",")) return PROFILE_NOMEM;
        add_comma = 1;
        if (!prof_run_text(p, "enum:")) return PROFILE_NOMEM;
        profile_status status = prof_dsl_value(p, member, 1);
        if (status != PROFILE_OK) return status;
    } else if (strcmp(upper, "ARRAY") == 0 &&
               (member = json_member(value, "items")) != NULL &&
               member->type == JSON_OBJECT && member->child) {
        if (add_comma && !prof_run_text(p, ",")) return PROFILE_NOMEM;
        add_comma = 1;
        profile_status status = prof_format_items(p, member);
        if (status != PROFILE_OK) return status;
    }
    member = json_member(value, "nullable");
    if (member && member->type == JSON_BOOL && member->boolean) {
        if (add_comma && !prof_run_text(p, ",")) return PROFILE_NOMEM;
        add_comma = 1;
        if (!prof_run_text(p, "nullable:true")) return PROFILE_NOMEM;
    }
    if (strcmp(upper, "OBJECT") == 0) {
        member = json_member(value, "properties");
        if (member && member->type == JSON_OBJECT) {
            if (add_comma && !prof_run_text(p, ",")) return PROFILE_NOMEM;
            add_comma = 1;
            if (!prof_run_text(p, "properties:{")) return PROFILE_NOMEM;
            profile_status status = prof_format_properties(p, member);
            if (status != PROFILE_OK) return status;
            if (!prof_run_text(p, "}")) return PROFILE_NOMEM;
        }
        member = json_member(value, "required");
        if (member && member->type == JSON_ARRAY) {
            if (add_comma && !prof_run_text(p, ",")) return PROFILE_NOMEM;
            add_comma = 1;
            if (!prof_run_text(p, "required:")) return PROFILE_NOMEM;
            profile_status status = prof_required_list(p, member);
            if (status != PROFILE_OK) return status;
        }
    }
    if (add_comma && !prof_run_text(p, ",")) return PROFILE_NOMEM;
    if (!prof_run_text(p, "type:") || !prof_quoted(p, upper, strlen(upper)) ||
        !prof_run_text(p, "}"))
        return PROFILE_NOMEM;
    return PROFILE_OK;
}

static profile_status prof_format_properties(profile *p,
                                             const json_value *properties) {
    size_t count;
    const json_value **sorted = prof_sorted(properties, &count);
    if (!sorted) return PROFILE_NOMEM;
    profile_status status = PROFILE_OK;
    for (size_t i = 0; i < count && status == PROFILE_OK; i++) {
        if (i && !prof_run_text(p, ",")) {
            status = PROFILE_NOMEM;
            break;
        }
        status = prof_format_property(p, sorted[i]->key, sorted[i]);
    }
    free(sorted);
    return status;
}

static profile_status prof_declaration(profile *p,
                                       const profile_tool *tool) {
    if (!tool->name || !tool->description) return PROFILE_INVALID_ARGUMENT;
    if (!prof_marker(p, PROFILE_ID_TOOL_BEGIN) ||
        !prof_run_text(p, "declaration:") ||
        !prof_run_text(p, tool->name) ||
        !prof_run_text(p, "{description:") ||
        !prof_quoted(p, tool->description, strlen(tool->description)))
        return PROFILE_NOMEM;
    if (tool->parameters_json && *tool->parameters_json) {
        json_value *params = json_parse(tool->parameters_json,
                                        strlen(tool->parameters_json));
        if (!params) return PROFILE_INVALID_ARGUMENT;
        profile_status status = PROFILE_OK;
        if (params->type != JSON_OBJECT) {
            status = PROFILE_INVALID_ARGUMENT;
        } else if (params->child) {
            char upper[24];
            const json_value *properties = json_member(params, "properties");
            const json_value *required = json_member(params, "required");
            if (!prof_type_upper(json_member(params, "type"), upper))
                status = PROFILE_INVALID_ARGUMENT;
            if (status == PROFILE_OK &&
                !prof_run_text(p, ",parameters:{"))
                status = PROFILE_NOMEM;
            if (status == PROFILE_OK && properties &&
                properties->type == JSON_OBJECT) {
                if (!prof_run_text(p, "properties:{"))
                    status = PROFILE_NOMEM;
                if (status == PROFILE_OK)
                    status = prof_format_properties(p, properties);
                if (status == PROFILE_OK && !prof_run_text(p, "},"))
                    status = PROFILE_NOMEM;
            }
            if (status == PROFILE_OK && required &&
                required->type == JSON_ARRAY) {
                if (!prof_run_text(p, "required:"))
                    status = PROFILE_NOMEM;
                if (status == PROFILE_OK)
                    status = prof_required_list(p, required);
                if (status == PROFILE_OK && !prof_run_text(p, ","))
                    status = PROFILE_NOMEM;
            }
            if (status == PROFILE_OK &&
                (!prof_run_text(p, "type:") ||
                 !prof_quoted(p, upper, strlen(upper)) ||
                 !prof_run_text(p, "}")))
                status = PROFILE_NOMEM;
        }
        json_free(params);
        if (status != PROFILE_OK) return status;
    }
    if (!prof_run_text(p, "}") || !prof_marker(p, PROFILE_ID_TOOL_END))
        return PROFILE_NOMEM;
    return PROFILE_OK;
}

profile_status profile_render_system(profile *p, const char *text,
                                     const profile_tool *tools,
                                     size_t tool_count,
                                     int thinking,
                                     profile_render *out) {
    if (!p || !out || (tool_count && !tools))
        return PROFILE_INVALID_ARGUMENT;
    prof_begin(p);
    if (!prof_marker(p, PROFILE_ID_BOS)) return PROFILE_NOMEM;
    const char *begin = "";
    size_t length = 0;
    if (text) prof_trim(text, &begin, &length);
    if (thinking || length || tool_count) {
        if (!prof_marker(p, PROFILE_ID_TURN_BEGIN) ||
            !prof_run_text(p, "system\n"))
            return PROFILE_NOMEM;
        if (thinking &&
            (!prof_marker(p, PROFILE_ID_THINK) ||
             !prof_run_text(p, "\n")))
            return PROFILE_NOMEM;
        if (!prof_run_put(p, begin, length)) return PROFILE_NOMEM;
        for (size_t i = 0; i < tool_count; i++) {
            profile_status status = prof_declaration(p, &tools[i]);
            if (status != PROFILE_OK) return status;
        }
        if (!prof_marker(p, PROFILE_ID_TURN_END) ||
            !prof_run_text(p, "\n"))
            return PROFILE_NOMEM;
    }
    return prof_finish(p, out);
}

static profile_status prof_prefix(profile *p, uint32_t turn) {
    switch (turn) {
    case PROFILE_TURN_PADDED:
        return PROFILE_OK;
    case PROFILE_TURN_BARE:
        return prof_run_text(p, "\n") ? PROFILE_OK : PROFILE_NOMEM;
    case PROFILE_TURN_OPEN:
        if (!prof_marker(p, PROFILE_ID_TURN_END) ||
            !prof_run_text(p, "\n"))
            return PROFILE_NOMEM;
        return PROFILE_OK;
    default:
        return PROFILE_INVALID_ARGUMENT;
    }
}

profile_status profile_render_user_image(profile *p, const char *text,
                                         uint32_t rows, uint32_t turn,
                                         profile_render *out, uint32_t *image_offset) {
    if (!p || !text || !out || !image_offset || rows < 1 || rows > XE_VISUAL_MAX)
        return PROFILE_INVALID_ARGUMENT;
    if (xe_token_id(p->engine, "<|image>") != PROFILE_ID_IMAGE_BEGIN ||
        xe_token_id(p->engine, "<|image|>") != PROFILE_ID_IMAGE ||
        xe_token_id(p->engine, "<image|>") != PROFILE_ID_IMAGE_END)
        return PROFILE_TEMPLATE_MISMATCH;
    prof_begin(p);
    profile_status status = prof_prefix(p, turn);
    if (status != PROFILE_OK) return status;
    if (!prof_marker(p, PROFILE_ID_TURN_BEGIN) || !prof_run_text(p, "user\n") ||
        !prof_marker(p, PROFILE_ID_IMAGE_BEGIN)) return PROFILE_NOMEM;
    *image_offset = p->token_count;
    for (uint32_t i = 0; i < rows; i++)
        if (!prof_marker(p, PROFILE_ID_IMAGE)) return PROFILE_NOMEM;
    const char *begin;
    size_t length;
    prof_trim(text, &begin, &length);
    if (!prof_marker(p, PROFILE_ID_IMAGE_END) || !prof_run_text(p, "\n") ||
        !prof_run_put(p, begin, length) || !prof_marker(p, PROFILE_ID_TURN_END) ||
        !prof_run_text(p, "\n")) return PROFILE_NOMEM;
    return prof_finish(p, out);
}

profile_status profile_render_user(profile *p, const char *text,
                                   uint32_t turn, profile_render *out) {
    if (!p || !out || !text) return PROFILE_INVALID_ARGUMENT;
    prof_begin(p);
    profile_status status = prof_prefix(p, turn);
    if (status != PROFILE_OK) return status;
    const char *begin;
    size_t length;
    prof_trim(text, &begin, &length);
    if (!prof_marker(p, PROFILE_ID_TURN_BEGIN) ||
        !prof_run_text(p, "user\n") ||
        !prof_run_put(p, begin, length) ||
        !prof_marker(p, PROFILE_ID_TURN_END) ||
        !prof_run_text(p, "\n"))
        return PROFILE_NOMEM;
    return prof_finish(p, out);
}

profile_status profile_render_tool_result(profile *p, const char *name,
                                          const char *value,
                                          profile_render *out) {
    if (!p || !out || !name || !value) return PROFILE_INVALID_ARGUMENT;
    prof_begin(p);
    if (!prof_marker(p, PROFILE_ID_RESPONSE_BEGIN) ||
        !prof_run_text(p, "response:") ||
        !prof_run_text(p, name) ||
        !prof_run_text(p, "{value:") ||
        !prof_quoted(p, value, strlen(value)) ||
        !prof_run_text(p, "}") ||
        !prof_marker(p, PROFILE_ID_RESPONSE_END))
        return PROFILE_NOMEM;
    return prof_finish(p, out);
}

profile_status profile_render_tool_image(profile *p,const char *name,const char *text,
                                         uint32_t rows,profile_render *out,uint32_t *offset) {
    if(!p||!name||!text||!out||!offset||rows<1||rows>XE_VISUAL_MAX)return PROFILE_INVALID_ARGUMENT;
    prof_begin(p);
    if(!prof_marker(p,PROFILE_ID_RESPONSE_BEGIN)||!prof_run_text(p,"response:")||
       !prof_run_text(p,name)||!prof_run_text(p,"{value:")||!prof_quoted(p,text,strlen(text))||
       !prof_run_text(p,",image:")||!prof_marker(p,PROFILE_ID_IMAGE_BEGIN))return PROFILE_NOMEM;
    *offset=p->token_count;
    for(uint32_t i=0;i<rows;i++)if(!prof_marker(p,PROFILE_ID_IMAGE))return PROFILE_NOMEM;
    if(!prof_marker(p,PROFILE_ID_IMAGE_END)||!prof_run_text(p,"}")||
       !prof_marker(p,PROFILE_ID_RESPONSE_END))return PROFILE_NOMEM;
    return prof_finish(p,out);
}

static profile_status prof_model_open(profile *p, uint32_t turn) {
    profile_status status = prof_prefix(p, turn);
    if (status != PROFILE_OK) return status;
    if (!prof_marker(p, PROFILE_ID_TURN_BEGIN) ||
        !prof_run_text(p, "model\n"))
        return PROFILE_NOMEM;
    return PROFILE_OK;
}

profile_status profile_render_assistant(profile *p, const char *reasoning,
                                        int reasoning_complete,
                                        const char *text,
                                        const profile_call *calls,
                                        size_t call_count, uint32_t turn,
                                        int turn_complete,
                                        profile_render *out) {
    if (!p || !out || (call_count && !calls))
        return PROFILE_INVALID_ARGUMENT;
    prof_begin(p);
    if (turn != PROFILE_TURN_OPEN) {
        profile_status status = prof_model_open(p, turn);
        if (status != PROFILE_OK) return status;
    }
    if (reasoning) {
        const char *begin;
        size_t length;
        prof_trim(reasoning, &begin, &length);
        if (!prof_marker(p, PROFILE_ID_CHANNEL_BEGIN) ||
            !prof_run_text(p, "thought\n") ||
            !prof_run_put(p, begin, length))
            return PROFILE_NOMEM;
        if (reasoning_complete &&
            (!prof_run_text(p, "\n") ||
             !prof_marker(p, PROFILE_ID_CHANNEL_END)))
            return PROFILE_NOMEM;
    }
    if (text) {
        const char *begin;
        size_t length;
        prof_trim(text, &begin, &length);
        if (!prof_run_put(p, begin, length)) return PROFILE_NOMEM;
    }
    for (size_t i = 0; i < call_count; i++) {
        if (!calls[i].name) return PROFILE_INVALID_ARGUMENT;
        if (!prof_marker(p, PROFILE_ID_CALL_BEGIN) ||
            !prof_run_text(p, "call:") ||
            !prof_run_text(p, calls[i].name))
            return PROFILE_NOMEM;
        if (calls[i].arguments_json && *calls[i].arguments_json) {
            json_value *arguments = json_parse(
                calls[i].arguments_json, strlen(calls[i].arguments_json));
            if (!arguments || arguments->type != JSON_OBJECT) {
                json_free(arguments);
                return PROFILE_INVALID_ARGUMENT;
            }
            profile_status status = prof_dsl_value(p, arguments, 0);
            json_free(arguments);
            if (status != PROFILE_OK) return status;
        } else {
            if (!prof_run_text(p, "{}")) return PROFILE_NOMEM;
        }
        if (!prof_marker(p, PROFILE_ID_CALL_END)) return PROFILE_NOMEM;
    }
    if (turn_complete && !call_count &&
        !prof_marker(p, PROFILE_ID_TURN_END))
        return PROFILE_NOMEM;
    return prof_finish(p, out);
}

profile_status profile_render_reply_open(profile *p, uint32_t turn,
                                         int thinking, int after_tool,
                                         profile_render *out) {
    if (!p || !out) return PROFILE_INVALID_ARGUMENT;
    prof_begin(p);
    if (turn != PROFILE_TURN_OPEN) {
        profile_status status = prof_model_open(p, turn);
        if (status != PROFILE_OK) return status;
    }
    if ((!thinking && turn != PROFILE_TURN_OPEN) ||
        (thinking && after_tool)) {
        if (!prof_marker(p, PROFILE_ID_CHANNEL_BEGIN) ||
            !prof_run_text(p, "thought\n"))
            return PROFILE_NOMEM;
        if (!thinking && !prof_marker(p, PROFILE_ID_CHANNEL_END))
            return PROFILE_NOMEM;
    }
    return prof_finish(p, out);
}

void profile_parser_reset(profile *p, int reasoning_open) {
    p->state = reasoning_open ? PROFILE_PS_THOUGHT : PROFILE_PS_CONTENT;
    p->in_string = 0;
    p->calls_done = 0;
    p->scan_length = 0;
    p->name_length = 0;
    p->depth = 0;
    json_writer_reset(&p->args);
}

size_t profile_parser_calls(const profile *p) {
    return p->calls_done;
}

int profile_parser_reasoning(const profile *p) {
    return p && p->state == PROFILE_PS_THOUGHT;
}

int32_t profile_reasoning_end_token(const profile *p) {
    (void)p;
    return PROFILE_ID_CHANNEL_END;
}

profile_status profile_get_reasoning_policy(
        const profile *p, uint32_t effort, int32_t budget_override,
        int32_t max_tokens, int32_t context_remaining,
        profile_reasoning_policy *out) {
    if (!p || !out || effort > PROFILE_REASONING_MAX ||
        budget_override < -1 || max_tokens < 0 || context_remaining < 0)
        return PROFILE_INVALID_ARGUMENT;
    int hard = -1;
    int window = 0;
    switch (effort) {
    case PROFILE_REASONING_LOW:
        hard = 128;
        window = 32;
        break;
    case PROFILE_REASONING_MEDIUM:
        hard = 512;
        window = 128;
        break;
    case PROFILE_REASONING_HIGH:
        hard = 2048;
        window = 512;
        break;
    case PROFILE_REASONING_MAX:
        hard = context_remaining;
        window = 512;
        break;
    default:
        break;
    }
    if (hard >= 0 && budget_override >= 0) hard = budget_override;
    if (hard >= 0) {
        int available = context_remaining;
        if (max_tokens > 0 && max_tokens < available) available = max_tokens;
        available = available > 256 ? available - 256 : 0;
        if (hard > available) hard = available;
    }
    out->hard_tokens = hard;
    out->soft_tokens = hard >= 0 && hard > window ? hard - window : 0;
    out->delimiter_rank = 3;
    out->delimiter_margin = 3.0f;
    return PROFILE_OK;
}

static int prof_scan_put(profile *p, const char *data, size_t length) {
    void *buffer = p->scan;
    if (!profile_grow(&buffer, &p->scan_capacity,
                      p->scan_length + length + 1, 1))
        return 0;
    p->scan = buffer;
    memcpy(p->scan + p->scan_length, data, length);
    p->scan_length += length;
    p->scan[p->scan_length] = '\0';
    return 1;
}

static int prof_name_put(profile *p, char c) {
    void *buffer = p->name;
    if (!profile_grow(&buffer, &p->name_capacity, p->name_length + 2, 1))
        return 0;
    p->name = buffer;
    p->name[p->name_length++] = c;
    p->name[p->name_length] = '\0';
    return 1;
}

static int prof_args_escape(profile *p, const unsigned char *data,
                            size_t length) {
    for (size_t i = 0; i < length; i++) {
        unsigned char c = data[i];
        if (c == '"' || c == '\\') {
            char pair[2] = { '\\', (char)c };
            if (!json_rawn(&p->args, pair, 2)) return 0;
        } else if (c == '\n') {
            if (!json_rawn(&p->args, "\\n", 2)) return 0;
        } else if (c == '\r') {
            if (!json_rawn(&p->args, "\\r", 2)) return 0;
        } else if (c == '\t') {
            if (!json_rawn(&p->args, "\\t", 2)) return 0;
        } else if (c < 0x20) {
            char escape[8];
            int n = snprintf(escape, sizeof escape, "\\u%04x", c);
            if (n != 6 || !json_rawn(&p->args, escape, 6)) return 0;
        } else {
            if (!json_rawn(&p->args, &c, 1)) return 0;
        }
    }
    return 1;
}

static void prof_call_drop(profile *p) {
    p->state = PROFILE_PS_CONTENT;
    p->in_string = 0;
    p->depth = 0;
    p->scan_length = 0;
    p->name_length = 0;
}

static int prof_args_pop(profile *p) {
    p->depth--;
    if (p->depth == 0) {
        p->state = PROFILE_PS_CALL_CLOSE;
        return 1;
    }
    p->level_phase[p->depth - 1] = PROFILE_ARG_AFTER;
    return 1;
}

static int prof_args_push(profile *p, int is_object) {
    if (p->depth >= PROFILE_MAX_DEPTH) return 0;
    p->level_kind[p->depth] = is_object ? 1 : 0;
    p->level_phase[p->depth] = is_object ? PROFILE_ARG_KEY
                                         : PROFILE_ARG_VALUE;
    p->depth++;
    return 1;
}

static int prof_args_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static int prof_args_byte(profile *p, unsigned char c) {
    int level = p->depth - 1;
    int object = p->level_kind[level];
    int phase = p->level_phase[level];
    if (object && phase == PROFILE_ARG_KEY) {
        if (prof_args_space(c) && !p->scan_length) return 1;
        if (c == '}' && !p->scan_length) {
            if (!json_rawn(&p->args, "}", 1)) return 0;
            return prof_args_pop(p);
        }
        if (c == ':') {
            if (!p->scan_length) return -1;
            if (!json_string(&p->args, p->scan, p->scan_length) ||
                !json_rawn(&p->args, ":", 1))
                return 0;
            p->scan_length = 0;
            p->level_phase[level] = PROFILE_ARG_VALUE;
            return 1;
        }
        char byte = (char)c;
        return prof_scan_put(p, &byte, 1);
    }
    if (phase == PROFILE_ARG_VALUE) {
        if (prof_args_space(c)) return 1;
        if (c == '{') {
            if (!json_rawn(&p->args, "{", 1)) return 0;
            return prof_args_push(p, 1) ? 1 : -1;
        }
        if (c == '[') {
            if (!json_rawn(&p->args, "[", 1)) return 0;
            return prof_args_push(p, 0) ? 1 : -1;
        }
        if (c == ']' && !object) {
            if (!json_rawn(&p->args, "]", 1)) return 0;
            return prof_args_pop(p);
        }
        if (c == ',' || c == '}' || c == ':') return -1;
        if (!json_rawn(&p->args, &c, 1)) return 0;
        p->level_phase[level] = PROFILE_ARG_BARE;
        return 1;
    }
    if (phase == PROFILE_ARG_BARE) {
        if (c == ',') {
            if (!json_rawn(&p->args, ",", 1)) return 0;
            p->level_phase[level] = object ? PROFILE_ARG_KEY
                                           : PROFILE_ARG_VALUE;
            return 1;
        }
        if (c == '}' && object) {
            if (!json_rawn(&p->args, "}", 1)) return 0;
            return prof_args_pop(p);
        }
        if (c == ']' && !object) {
            if (!json_rawn(&p->args, "]", 1)) return 0;
            return prof_args_pop(p);
        }
        if (prof_args_space(c)) {
            p->level_phase[level] = PROFILE_ARG_AFTER;
            return 1;
        }
        if (c == '{' || c == '[' || c == ':') return -1;
        if (!json_rawn(&p->args, &c, 1)) return 0;
        return 1;
    }
    if (prof_args_space(c)) return 1;
    if (c == ',') {
        if (!json_rawn(&p->args, ",", 1)) return 0;
        p->level_phase[level] = object ? PROFILE_ARG_KEY
                                       : PROFILE_ARG_VALUE;
        return 1;
    }
    if (c == '}' && object) {
        if (!json_rawn(&p->args, "}", 1)) return 0;
        return prof_args_pop(p);
    }
    if (c == ']' && !object) {
        if (!json_rawn(&p->args, "]", 1)) return 0;
        return prof_args_pop(p);
    }
    return -1;
}

static profile_status prof_stop(profile *p, profile_parse_event *out,
                                uint32_t reason, int include) {
    p->state = PROFILE_PS_DONE;
    out->kind = PROFILE_PARSE_STOP;
    out->stop_reason = reason;
    out->include_token = include;
    return PROFILE_OK;
}

profile_status profile_parser_feed(profile *p, int32_t token,
                                   profile_parse_event *out) {
    if (!p || !out) return PROFILE_INVALID_ARGUMENT;
    memset(out, 0, sizeof *out);
    out->kind = PROFILE_PARSE_NONE;
    out->include_token = 1;
    if (p->state == PROFILE_PS_DONE) return PROFILE_INVALID_ARGUMENT;

    if (token == PROFILE_ID_TURN_END)
        return prof_stop(p, out, PROFILE_STOP_EOT, 1);
    if (token == PROFILE_ID_EOS)
        return prof_stop(p, out, PROFILE_STOP_EOS, 1);
    if (token == PROFILE_ID_RESPONSE_BEGIN)
        return prof_stop(p, out, PROFILE_STOP_TOOL_CALLS, 0);

    int detok = xe_detokenize(p->engine, token, p->detok,
                              (int)sizeof p->detok);
    if (detok < 0) detok = 0;

    switch (p->state) {
    case PROFILE_PS_CONTENT:
        if (token == PROFILE_ID_CALL_BEGIN) {
            p->state = PROFILE_PS_NAME;
            p->scan_length = 0;
            p->name_length = 0;
            return PROFILE_OK;
        }
        if (token == PROFILE_ID_CHANNEL_BEGIN) {
            p->state = PROFILE_PS_CHANNEL;
            p->scan_length = 0;
            return PROFILE_OK;
        }
        if (token == PROFILE_ID_CHANNEL_END) return PROFILE_OK;
        if (detok > 0) {
            out->kind = PROFILE_PARSE_TEXT;
            out->text = (const uint8_t *)p->detok;
            out->text_length = (size_t)detok;
        }
        return PROFILE_OK;
    case PROFILE_PS_CHANNEL: {
        if (token == PROFILE_ID_CHANNEL_END) {
            p->state = PROFILE_PS_CONTENT;
            p->scan_length = 0;
            return PROFILE_OK;
        }
        if (token == PROFILE_ID_CALL_BEGIN) {
            p->state = PROFILE_PS_NAME;
            p->scan_length = 0;
            p->name_length = 0;
            return PROFILE_OK;
        }
        if (detok > 0 && !prof_scan_put(p, p->detok, (size_t)detok))
            return PROFILE_NOMEM;
        static const char header[] = "thought\n";
        size_t header_length = sizeof header - 1;
        size_t match = p->scan_length < header_length ? p->scan_length
                                                      : header_length;
        if (memcmp(p->scan, header, match) != 0) {
            p->state = PROFILE_PS_CONTENT;
            out->kind = PROFILE_PARSE_TEXT;
            out->text = (const uint8_t *)p->scan;
            out->text_length = p->scan_length;
            return PROFILE_OK;
        }
        if (p->scan_length >= header_length) {
            p->state = PROFILE_PS_THOUGHT;
            size_t remaining = p->scan_length - header_length;
            if (remaining) {
                memmove(p->scan, p->scan + header_length, remaining);
                p->scan_length = remaining;
                out->kind = PROFILE_PARSE_REASONING;
                out->text = (const uint8_t *)p->scan;
                out->text_length = remaining;
            } else {
                p->scan_length = 0;
            }
        }
        return PROFILE_OK;
    }
    case PROFILE_PS_THOUGHT:
        if (token == PROFILE_ID_CHANNEL_END) {
            p->state = PROFILE_PS_CONTENT;
        } else if (detok > 0) {
            out->kind = PROFILE_PARSE_REASONING;
            out->text = (const uint8_t *)p->detok;
            out->text_length = (size_t)detok;
        }
        return PROFILE_OK;
    case PROFILE_PS_NAME: {
        static const char keyword[] = "call:";
        size_t keyword_length = sizeof keyword - 1;
        for (int i = 0; i < detok; i++) {
            char c = p->detok[i];
            if (p->scan_length < keyword_length) {
                if (c != keyword[p->scan_length]) {
                    prof_call_drop(p);
                    return PROFILE_OK;
                }
                if (!prof_scan_put(p, &c, 1)) return PROFILE_NOMEM;
                continue;
            }
            if (c == '{') {
                if (!p->name_length) {
                    prof_call_drop(p);
                    return PROFILE_OK;
                }
                p->state = PROFILE_PS_ARGS;
                p->in_string = 0;
                p->depth = 0;
                json_writer_reset(&p->args);
                if (!json_rawn(&p->args, "{", 1)) return PROFILE_NOMEM;
                if (!prof_args_push(p, 1)) return PROFILE_NOMEM;
                p->scan_length = 0;
                for (int j = i + 1; j < detok; j++) {
                    unsigned char byte = (unsigned char)p->detok[j];
                    if (p->state == PROFILE_PS_CALL_CLOSE) {
                        if (prof_args_space(byte)) continue;
                        prof_call_drop(p);
                        return PROFILE_OK;
                    }
                    int fed = prof_args_byte(p, byte);
                    if (!fed) return PROFILE_NOMEM;
                    if (fed < 0) {
                        prof_call_drop(p);
                        return PROFILE_OK;
                    }
                }
                out->kind = PROFILE_PARSE_CALL_START;
                out->call_name = p->name;
                return PROFILE_OK;
            }
            if (!prof_name_put(p, c)) return PROFILE_NOMEM;
        }
        if (token == PROFILE_ID_CALL_BEGIN ||
            token == PROFILE_ID_CALL_END ||
            token == PROFILE_ID_QUOTE ||
            token == PROFILE_ID_CHANNEL_BEGIN ||
            token == PROFILE_ID_CHANNEL_END ||
            token == PROFILE_ID_TOOL_BEGIN ||
            token == PROFILE_ID_TOOL_END ||
            token == PROFILE_ID_RESPONSE_END)
            prof_call_drop(p);
        return PROFILE_OK;
    }
    case PROFILE_PS_ARGS:
        if (token == PROFILE_ID_QUOTE) {
            if (p->in_string) {
                p->in_string = 0;
                if (!json_rawn(&p->args, "\"", 1)) return PROFILE_NOMEM;
                p->level_phase[p->depth - 1] = PROFILE_ARG_AFTER;
                return PROFILE_OK;
            }
            int level = p->depth - 1;
            if (p->level_kind[level] &&
                p->level_phase[level] == PROFILE_ARG_KEY)
                return PROFILE_OK;
            if (p->level_phase[level] != PROFILE_ARG_VALUE) {
                prof_call_drop(p);
                return PROFILE_OK;
            }
            p->in_string = 1;
            if (!json_rawn(&p->args, "\"", 1)) return PROFILE_NOMEM;
            return PROFILE_OK;
        }
        if (p->in_string) {
            if (token == PROFILE_ID_CALL_BEGIN ||
                token == PROFILE_ID_CALL_END ||
                token == PROFILE_ID_CHANNEL_BEGIN ||
                token == PROFILE_ID_CHANNEL_END ||
                token == PROFILE_ID_TOOL_BEGIN ||
                token == PROFILE_ID_TOOL_END ||
                token == PROFILE_ID_RESPONSE_END ||
                token == PROFILE_ID_TURN_BEGIN) {
                prof_call_drop(p);
                return PROFILE_OK;
            }
            if (detok > 0 &&
                !prof_args_escape(p, (const unsigned char *)p->detok,
                                  (size_t)detok))
                return PROFILE_NOMEM;
            return PROFILE_OK;
        }
        if (token == PROFILE_ID_CALL_END) {
            prof_call_drop(p);
            return PROFILE_OK;
        }
        for (int i = 0; i < detok; i++) {
            unsigned char byte = (unsigned char)p->detok[i];
            if (p->state == PROFILE_PS_CALL_CLOSE) {
                if (prof_args_space(byte)) continue;
                prof_call_drop(p);
                return PROFILE_OK;
            }
            int fed = prof_args_byte(p, byte);
            if (!fed) return PROFILE_NOMEM;
            if (fed < 0) {
                prof_call_drop(p);
                return PROFILE_OK;
            }
        }
        return PROFILE_OK;
    case PROFILE_PS_CALL_CLOSE:
        if (token == PROFILE_ID_CALL_END) {
            p->state = PROFILE_PS_CONTENT;
            p->calls_done++;
            out->kind = PROFILE_PARSE_CALL_END;
            out->call_name = p->name;
            out->arguments_json = p->args.data ? p->args.data : "{}";
            return PROFILE_OK;
        }
        prof_call_drop(p);
        if (detok > 0) {
            out->kind = PROFILE_PARSE_TEXT;
            out->text = (const uint8_t *)p->detok;
            out->text_length = (size_t)detok;
        }
        return PROFILE_OK;
    default:
        return PROFILE_INVALID_ARGUMENT;
    }
}
