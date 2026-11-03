#include "json.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    JSON_MAX_DEPTH = 64
};

typedef struct {
    const char *p;
    const char *end;
    int depth;
} json_parser;

static int json_utf8_width(unsigned char lead) {
    if (lead >= 0xc2 && lead <= 0xdf) return 2;
    if (lead >= 0xe0 && lead <= 0xef) return 3;
    if (lead >= 0xf0 && lead <= 0xf4) return 4;
    return 0;
}
static int json_utf8_prefix(const unsigned char *p, size_t n) {
    int width = json_utf8_width(p[0]);
    if (!width || n > (size_t)width) return 0;
    for (size_t i=1;i<n;i++) if ((p[i]&0xc0)!=0x80) return 0;
    if (n>1 && ((p[0]==0xe0 && p[1]<0xa0) || (p[0]==0xed && p[1]>=0xa0) ||
                (p[0]==0xf0 && p[1]<0x90) || (p[0]==0xf4 && p[1]>=0x90))) return 0;
    return 1;
}
static void json_skip_space(json_parser *ps) {
    while (ps->p < ps->end &&
           (*ps->p == ' ' || *ps->p == '\t' || *ps->p == '\n' ||
            *ps->p == '\r'))
        ps->p++;
}

static json_value *json_value_new(uint32_t type) {
    json_value *v = calloc(1, sizeof *v);
    if (v) v->type = type;
    return v;
}

void json_free(json_value *value) {
    while (value) {
        json_value *next = value->next;
        json_free(value->child);
        free(value->text);
        free(value->key);
        free(value);
        value = next;
    }
}

static int json_append_byte(char **data, size_t *length, size_t *capacity,
                            char byte) {
    if (*length + 1 >= *capacity) {
        size_t grown = *capacity ? *capacity * 2 : 32;
        char *p = realloc(*data, grown);
        if (!p) return 0;
        *data = p;
        *capacity = grown;
    }
    (*data)[(*length)++] = byte;
    return 1;
}

static int json_append_utf8(char **data, size_t *length, size_t *capacity,
                            uint32_t code) {
    char bytes[4];
    int n;
    if (code < 0x80) {
        bytes[0] = (char)code;
        n = 1;
    } else if (code < 0x800) {
        bytes[0] = (char)(0xc0 | code >> 6);
        bytes[1] = (char)(0x80 | (code & 0x3f));
        n = 2;
    } else if (code < 0x10000) {
        bytes[0] = (char)(0xe0 | code >> 12);
        bytes[1] = (char)(0x80 | (code >> 6 & 0x3f));
        bytes[2] = (char)(0x80 | (code & 0x3f));
        n = 3;
    } else {
        bytes[0] = (char)(0xf0 | code >> 18);
        bytes[1] = (char)(0x80 | (code >> 12 & 0x3f));
        bytes[2] = (char)(0x80 | (code >> 6 & 0x3f));
        bytes[3] = (char)(0x80 | (code & 0x3f));
        n = 4;
    }
    for (int i = 0; i < n; i++)
        if (!json_append_byte(data, length, capacity, bytes[i])) return 0;
    return 1;
}

static int json_hex4(json_parser *ps, uint32_t *out) {
    if (ps->end - ps->p < 4) return 0;
    uint32_t value = 0;
    for (int i = 0; i < 4; i++) {
        char c = ps->p[i];
        value <<= 4;
        if (c >= '0' && c <= '9') value |= (uint32_t)(c - '0');
        else if (c >= 'a' && c <= 'f') value |= (uint32_t)(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') value |= (uint32_t)(c - 'A' + 10);
        else return 0;
    }
    ps->p += 4;
    *out = value;
    return 1;
}

static int json_parse_string(json_parser *ps, char **out,
                             size_t *out_length) {
    if (ps->p >= ps->end || *ps->p != '"') return 0;
    ps->p++;
    char *data = NULL;
    size_t length = 0, capacity = 0;
    while (ps->p < ps->end && *ps->p != '"') {
        unsigned char c = (unsigned char)*ps->p;
        if (c < 0x20) goto fail;
        if (c == '\\') {
            ps->p++;
            if (ps->p >= ps->end) goto fail;
            char e = *ps->p++;
            char plain;
            switch (e) {
            case '"': plain = '"'; break;
            case '\\': plain = '\\'; break;
            case '/': plain = '/'; break;
            case 'b': plain = '\b'; break;
            case 'f': plain = '\f'; break;
            case 'n': plain = '\n'; break;
            case 'r': plain = '\r'; break;
            case 't': plain = '\t'; break;
            case 'u': {
                uint32_t code;
                if (!json_hex4(ps, &code)) goto fail;
                if (code >= 0xd800 && code <= 0xdbff) {
                    if (ps->end - ps->p < 2 || ps->p[0] != '\\' ||
                        ps->p[1] != 'u') goto fail;
                    ps->p += 2;
                    uint32_t low;
                    if (!json_hex4(ps, &low) ||
                        low < 0xdc00 || low > 0xdfff) goto fail;
                    code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
                } else if (code >= 0xdc00 && code <= 0xdfff) {
                    goto fail;
                }
                if (code == 0) goto fail;
                if (!json_append_utf8(&data, &length, &capacity, code))
                    goto fail;
                continue;
            }
            default: goto fail;
            }
            if (!json_append_byte(&data, &length, &capacity, plain))
                goto fail;
        } else if (c >= 0x80) {
            int width=json_utf8_width(c);
            if(!width || ps->end-ps->p<width ||
               !json_utf8_prefix((const unsigned char *)ps->p,(size_t)width)) goto fail;
            for(int i=0;i<width;i++)
                if(!json_append_byte(&data,&length,&capacity,ps->p[i]))goto fail;
            ps->p+=width;
        } else {
            if (!json_append_byte(&data, &length, &capacity, (char)c))
                goto fail;
            ps->p++;
        }
    }
    if (ps->p >= ps->end) goto fail;
    ps->p++;
    if (!data) {
        data = malloc(1);
        if (!data) return 0;
    } else {
        char *shrunk = realloc(data, length + 1);
        if (shrunk) data = shrunk;
    }
    data[length] = '\0';
    *out = data;
    *out_length = length;
    return 1;
fail:
    free(data);
    return 0;
}

static json_value *json_parse_value(json_parser *ps);

static json_value *json_parse_container(json_parser *ps, uint32_t type) {
    char open = type == JSON_OBJECT ? '{' : '[';
    char close = type == JSON_OBJECT ? '}' : ']';
    if (ps->depth >= JSON_MAX_DEPTH || *ps->p != open) return NULL;
    ps->p++;
    ps->depth++;
    json_value *container = json_value_new(type);
    if (!container) goto fail;
    json_value **tail = &container->child;
    json_skip_space(ps);
    if (ps->p < ps->end && *ps->p == close) {
        ps->p++;
        ps->depth--;
        return container;
    }
    for (;;) {
        json_skip_space(ps);
        char *key = NULL;
        size_t key_length = 0;
        if (type == JSON_OBJECT) {
            if (!json_parse_string(ps, &key, &key_length)) goto fail;
            if (memchr(key, 0, key_length)) {
                free(key);
                goto fail;
            }
            json_skip_space(ps);
            if (ps->p >= ps->end || *ps->p != ':') {
                free(key);
                goto fail;
            }
            ps->p++;
        }
        json_value *item = json_parse_value(ps);
        if (!item) {
            free(key);
            goto fail;
        }
        item->key = key;
        *tail = item;
        tail = &item->next;
        json_skip_space(ps);
        if (ps->p >= ps->end) goto fail;
        if (*ps->p == ',') {
            ps->p++;
            continue;
        }
        if (*ps->p == close) {
            ps->p++;
            ps->depth--;
            return container;
        }
        goto fail;
    }
fail:
    json_free(container);
    return NULL;
}

static json_value *json_parse_value(json_parser *ps) {
    json_skip_space(ps);
    if (ps->p >= ps->end) return NULL;
    char c = *ps->p;
    if (c == '{') return json_parse_container(ps, JSON_OBJECT);
    if (c == '[') return json_parse_container(ps, JSON_ARRAY);
    if (c == '"') {
        json_value *v = json_value_new(JSON_STRING);
        if (!v) return NULL;
        if (!json_parse_string(ps, &v->text, &v->text_length)) {
            json_free(v);
            return NULL;
        }
        return v;
    }
    if (c == 't' || c == 'f') {
        size_t n = c == 't' ? 4 : 5;
        const char *word = c == 't' ? "true" : "false";
        if ((size_t)(ps->end - ps->p) < n ||
            memcmp(ps->p, word, n) != 0) return NULL;
        ps->p += n;
        json_value *v = json_value_new(JSON_BOOL);
        if (v) v->boolean = c == 't';
        return v;
    }
    if (c == 'n') {
        if (ps->end - ps->p < 4 || memcmp(ps->p, "null", 4) != 0)
            return NULL;
        ps->p += 4;
        return json_value_new(JSON_NULL);
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
        const char *start = ps->p;
        if (*ps->p == '-') ps->p++;
        if (ps->p >= ps->end) return NULL;
        if (*ps->p == '0') {
            ps->p++;
        } else if (*ps->p >= '1' && *ps->p <= '9') {
            while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9')
                ps->p++;
        } else {
            return NULL;
        }
        if (ps->p < ps->end && *ps->p == '.') {
            ps->p++;
            if (ps->p >= ps->end || *ps->p < '0' || *ps->p > '9')
                return NULL;
            while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9')
                ps->p++;
        }
        if (ps->p < ps->end && (*ps->p == 'e' || *ps->p == 'E')) {
            ps->p++;
            if (ps->p < ps->end && (*ps->p == '+' || *ps->p == '-'))
                ps->p++;
            if (ps->p >= ps->end || *ps->p < '0' || *ps->p > '9')
                return NULL;
            while (ps->p < ps->end && *ps->p >= '0' && *ps->p <= '9')
                ps->p++;
        }
        size_t raw_length = (size_t)(ps->p - start);
        json_value *v = json_value_new(JSON_NUMBER);
        if (!v) return NULL;
        v->text = malloc(raw_length + 1);
        if (!v->text) {
            json_free(v);
            return NULL;
        }
        memcpy(v->text, start, raw_length);
        v->text[raw_length] = '\0';
        v->text_length = raw_length;
        v->number = strtod(v->text, NULL);
        return v;
    }
    return NULL;
}

json_value *json_parse(const char *data, size_t length) {
    if (!data) return NULL;
    json_parser ps = { data, data + length, 0 };
    json_value *v = json_parse_value(&ps);
    if (!v) return NULL;
    json_skip_space(&ps);
    if (ps.p != ps.end) {
        json_free(v);
        return NULL;
    }
    return v;
}

const json_value *json_member(const json_value *object, const char *key) {
    if (!object || object->type != JSON_OBJECT) return NULL;
    for (const json_value *v = object->child; v; v = v->next)
        if (v->key && strcmp(v->key, key) == 0) return v;
    return NULL;
}

const json_value *json_index(const json_value *array, size_t index) {
    if (!array || array->type != JSON_ARRAY) return NULL;
    const json_value *v = array->child;
    while (v && index--) v = v->next;
    return v;
}

size_t json_length(const json_value *value) {
    size_t n = 0;
    if (value)
        for (const json_value *v = value->child; v; v = v->next) n++;
    return n;
}

void json_writer_reset(json_writer *w) {
    w->length = 0;
    w->failed = 0;
    if (w->data) w->data[0] = '\0';
}

void json_writer_free(json_writer *w) {
    free(w->data);
    w->data = NULL;
    w->length = 0;
    w->capacity = 0;
    w->failed = 0;
}

static int json_writer_reserve(json_writer *w, size_t extra) {
    if (w->failed) return 0;
    if (extra >= SIZE_MAX - w->length) {
        w->failed = 1;
        return 0;
    }
    size_t need = w->length + extra + 1;
    if (need <= w->capacity) return 1;
    size_t capacity = w->capacity ? w->capacity : 64;
    while (capacity < need) {
        if (capacity > SIZE_MAX / 2) {
            w->failed = 1;
            return 0;
        }
        capacity *= 2;
    }
    char *grown = realloc(w->data, capacity);
    if (!grown) {
        w->failed = 1;
        return 0;
    }
    w->data = grown;
    w->capacity = capacity;
    return 1;
}

int json_rawn(json_writer *w, const void *data, size_t length) {
    if (!json_writer_reserve(w, length)) return 0;
    memcpy(w->data + w->length, data, length);
    w->length += length;
    w->data[w->length] = '\0';
    return 1;
}

int json_raw(json_writer *w, const char *text) {
    return json_rawn(w, text, strlen(text));
}

size_t json_utf8_incomplete_tail(const char *text, size_t length) {
    const unsigned char *p=(const unsigned char *)text;
    size_t first=length>3?length-3:0;
    for(size_t i=first;i<length;i++) {
        int width=json_utf8_width(p[i]);
        if(width && length-i<(size_t)width && json_utf8_prefix(p+i,length-i))return length-i;
    }
    return 0;
}

int json_string(json_writer *w, const char *text, size_t length) {
    if (!json_rawn(w, "\"", 1)) return 0;
    const unsigned char *p = (const unsigned char *)text;
    for (size_t i = 0; i < length; i++) {
        unsigned char c = p[i];
        if (c == '"' || c == '\\') {
            char pair[2] = { '\\', (char)c };
            if (!json_rawn(w, pair, 2)) return 0;
        } else if (c == '\n') {
            if (!json_rawn(w, "\\n", 2)) return 0;
        } else if (c == '\r') {
            if (!json_rawn(w, "\\r", 2)) return 0;
        } else if (c == '\t') {
            if (!json_rawn(w, "\\t", 2)) return 0;
        } else if (c < 0x20) {
            char escape[8];
            snprintf(escape, sizeof escape, "\\u%04x", c);
            if (!json_rawn(w, escape, 6)) return 0;
        } else if (c >= 0x80) {
            int width=json_utf8_width(c);
            size_t available=length-i;
            if(width && available>=(size_t)width && json_utf8_prefix(p+i,(size_t)width)) {
                if(!json_rawn(w,p+i,(size_t)width))return 0;
                i+=(size_t)width-1;
            } else {
                if(!json_raw(w,"\\ufffd"))return 0;
                if(width && available<(size_t)width && json_utf8_prefix(p+i,available))i=length-1;
            }
        } else {
            if (!json_rawn(w, &c, 1)) return 0;
        }
    }
    return json_rawn(w, "\"", 1);
}

int json_u64(json_writer *w, uint64_t value) {
    char text[32];
    int n = snprintf(text, sizeof text, "%llu", (unsigned long long)value);
    return n > 0 && json_rawn(w, text, (size_t)n);
}

int json_i64(json_writer *w, int64_t value) {
    char text[32];
    int n = snprintf(text, sizeof text, "%lld", (long long)value);
    return n > 0 && json_rawn(w, text, (size_t)n);
}

int json_double(json_writer *w, double value) {
    if (!isfinite(value)) return json_rawn(w, "null", 4);
    char text[40];
    int n = snprintf(text, sizeof text, "%.17g", value);
    return n > 0 && json_rawn(w, text, (size_t)n);
}

int json_write_value(json_writer *w, const json_value *value) {
    if (!value) return json_rawn(w, "null", 4);
    switch (value->type) {
    case JSON_NULL:
        return json_rawn(w, "null", 4);
    case JSON_BOOL:
        return value->boolean ? json_rawn(w, "true", 4)
                              : json_rawn(w, "false", 5);
    case JSON_NUMBER:
        return json_rawn(w, value->text, value->text_length);
    case JSON_STRING:
        return json_string(w, value->text, value->text_length);
    case JSON_ARRAY: {
        if (!json_rawn(w, "[", 1)) return 0;
        int first = 1;
        for (const json_value *v = value->child; v; v = v->next) {
            if (!first && !json_rawn(w, ",", 1)) return 0;
            first = 0;
            if (!json_write_value(w, v)) return 0;
        }
        return json_rawn(w, "]", 1);
    }
    case JSON_OBJECT: {
        if (!json_rawn(w, "{", 1)) return 0;
        int first = 1;
        for (const json_value *v = value->child; v; v = v->next) {
            if (!first && !json_rawn(w, ",", 1)) return 0;
            first = 0;
            if (!json_string(w, v->key, strlen(v->key)) ||
                !json_rawn(w, ":", 1) ||
                !json_write_value(w, v))
                return 0;
        }
        return json_rawn(w, "}", 1);
    }
    default:
        return 0;
    }
}
