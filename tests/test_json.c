#include "../json.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition) do { \
    if (!(condition)) { \
        failures++; \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                #condition); \
    } \
} while (0)

static json_value *parse(const char *text) {
    return json_parse(text, strlen(text));
}

int main(void) {
    json_value *v = parse("{\"a\":1,\"b\":[true,false,null],"
                          "\"c\":{\"x\":\"y\"},\"d\":-1.5e3}");
    CHECK(v && v->type == JSON_OBJECT);
    CHECK(json_length(v) == 4);
    const json_value *a = json_member(v, "a");
    CHECK(a && a->type == JSON_NUMBER && a->number == 1.0);
    CHECK(strcmp(a->text, "1") == 0);
    const json_value *b = json_member(v, "b");
    CHECK(b && b->type == JSON_ARRAY && json_length(b) == 3);
    CHECK(json_index(b, 0)->type == JSON_BOOL &&
          json_index(b, 0)->boolean == 1);
    CHECK(json_index(b, 2)->type == JSON_NULL);
    CHECK(json_index(b, 3) == NULL);
    const json_value *c = json_member(v, "c");
    CHECK(c && c->type == JSON_OBJECT);
    const json_value *x = json_member(c, "x");
    CHECK(x && x->type == JSON_STRING && strcmp(x->text, "y") == 0);
    const json_value *d = json_member(v, "d");
    CHECK(d && d->number == -1500.0);

    json_writer w = {0};
    CHECK(json_write_value(&w, v));
    json_value *round = json_parse(w.data, w.length);
    CHECK(round != NULL);
    json_writer w2 = {0};
    CHECK(json_write_value(&w2, round));
    CHECK(w.length == w2.length &&
          memcmp(w.data, w2.data, w.length) == 0);
    json_writer_free(&w2);
    json_writer_free(&w);
    json_free(round);
    json_free(v);

    v = parse("\"a\\\"b\\\\c\\/d\\b\\f\\n\\r\\t\\u0041\\u00e8"
              "\\ud83d\\ude00\"");
    CHECK(v && v->type == JSON_STRING);
    CHECK(strcmp(v->text, "a\"b\\c/d\b\f\n\r\tA\xc3\xa8\xf0\x9f\x98\x80")
          == 0);
    json_free(v);

    v = parse("  [ 1 , 2 ]  ");
    CHECK(v && v->type == JSON_ARRAY && json_length(v) == 2);
    json_free(v);

    CHECK(parse("") == NULL);
    CHECK(parse("{") == NULL);
    CHECK(parse("{\"a\":}") == NULL);
    CHECK(parse("[1,]") == NULL);
    CHECK(parse("01") == NULL);
    CHECK(parse("1.") == NULL);
    CHECK(parse("\"\\u0000\"") == NULL);
    CHECK(parse("\"\\ud800\"") == NULL);
    CHECK(parse("nul") == NULL);
    CHECK(parse("{\"a\":1}x") == NULL);
    CHECK(parse("{\"a\":1,\"a\":2}") != NULL);

    char deep[300];
    memset(deep, '[', 150);
    memset(deep + 150, ']', 150);
    CHECK(json_parse(deep, 300) == NULL);

    CHECK(parse("\"\xff\"")==NULL);
    CHECK(parse("\"\xed\xa0\x80\"")==NULL);
    CHECK(parse("\"\xf4\x90\x80\x80\"")==NULL);
    CHECK(parse("\"\xc2\"")==NULL);
    v=parse("\"\xf0\x9f\x98\x80\"");CHECK(v && v->text_length==4);json_free(v);
    CHECK(json_utf8_incomplete_tail("abc",3)==0);
    CHECK(json_utf8_incomplete_tail("\xc2",1)==1);
    CHECK(json_utf8_incomplete_tail("\xe2\x82",2)==2);
    CHECK(json_utf8_incomplete_tail("\xf0\x9f\x98",3)==3);
    CHECK(json_utf8_incomplete_tail("\xc2\xa0",2)==0);
    CHECK(json_utf8_incomplete_tail("\xed\xa0",2)==0);
    json_writer utf8={0};
    CHECK(json_string(&utf8,"\xc2\xa0",2));
    CHECK(utf8.length==4&&!memcmp(utf8.data,"\"\xc2\xa0\"",4));
    json_writer_reset(&utf8);CHECK(json_string(&utf8,"\xe2\x82",2));
    CHECK(!strcmp(utf8.data,"\"\\ufffd\""));
    json_writer_reset(&utf8);CHECK(json_string(&utf8,"\xff",1));
    CHECK(!strcmp(utf8.data,"\"\\ufffd\""));json_writer_free(&utf8);
    json_writer out = {0};
    CHECK(json_raw(&out, "{\"s\":"));
    CHECK(json_string(&out, "a\"b\\c\nd\x01", 8));
    CHECK(json_raw(&out, ",\"n\":"));
    CHECK(json_u64(&out, UINT64_C(18446744073709551615)));
    CHECK(json_raw(&out, ",\"i\":"));
    CHECK(json_i64(&out, INT64_C(-42)));
    CHECK(json_raw(&out, ",\"f\":"));
    CHECK(json_double(&out, 0.5));
    CHECK(json_raw(&out, "}"));
    v = json_parse(out.data, out.length);
    CHECK(v != NULL);
    const json_value *s = json_member(v, "s");
    CHECK(s && s->text_length == 8 &&
          memcmp(s->text, "a\"b\\c\nd\x01", 8) == 0);
    CHECK(json_member(v, "n")->number ==
          (double)UINT64_C(18446744073709551615));
    CHECK(json_member(v, "i")->number == -42.0);
    CHECK(json_member(v, "f")->number == 0.5);
    json_free(v);
    json_writer_free(&out);

    if (failures) {
        fprintf(stderr, "test_json: %d failures\n", failures);
        return 1;
    }
    printf("test_json: all checks passed\n");
    return 0;
}
