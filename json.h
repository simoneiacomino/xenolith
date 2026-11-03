#ifndef JSON_H
#define JSON_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} json_type;

typedef struct json_value json_value;

struct json_value {
    uint32_t type;
    int boolean;
    double number;
    char *text;
    size_t text_length;
    char *key;
    json_value *child;
    json_value *next;
};

json_value *json_parse(const char *data, size_t length);
void json_free(json_value *value);
const json_value *json_member(const json_value *object, const char *key);
const json_value *json_index(const json_value *array, size_t index);
size_t json_length(const json_value *value);

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
    int failed;
} json_writer;

void json_writer_reset(json_writer *w);
void json_writer_free(json_writer *w);
int json_raw(json_writer *w, const char *text);
int json_rawn(json_writer *w, const void *data, size_t length);
/* Trailing bytes of a valid, incomplete UTF-8 sequence (zero when complete). */
size_t json_utf8_incomplete_tail(const char *text, size_t length);
int json_string(json_writer *w, const char *text, size_t length);
int json_u64(json_writer *w, uint64_t value);
int json_i64(json_writer *w, int64_t value);
int json_double(json_writer *w, double value);
int json_write_value(json_writer *w, const json_value *value);

#endif
