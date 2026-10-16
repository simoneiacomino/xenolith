/* Real snapshot/cache code with deterministic host-backed KV. No model or
 * encoder is opened: this checks storage, image identity and retention. */
#define SNAPSHOT_FIXTURE_NO_MAIN
#include "snapshot_compat_fixture.c"
#include "../kvstore.h"
#include <dirent.h>

static void attach(xe_session *s, xe_visual_span *span, int n) {
    fixture_state(s,n,0,0);
    xe_input input={s->tokens,n,span,1};
    xe_visual_context_release(s->visual_context);
    s->visual_context=xe_visual_context_new(&input);CHECK(s->visual_context);
    s->visual_prompt_end=n;
}
static kvstore_id save(kvstore *store,xe_session *s,int n) {
    kvstore_save_options options={.has_group=1,.rebuild_cost=(uint64_t)n};
    options.group.bytes[0]=1;
    kvstore_id id;CHECK(kvstore_save(store,s,&options,&id,NULL)==KVSTORE_OK);return id;
}
static void rewrite_header(FILE *f,uint8_t h[384]) {
    format_put_u64le(h+376,format_crc64(0,h,376));
    CHECK(format_seek(f,0)&&format_write(f,h,384)&&fflush(f)==0);
}
int main(void) {
    uint8_t dummy=0,media=1;
    xe_engine e={.context=2049,.pool_initialized=1,.owner=pthread_self(),.scalar_rms=1,
        .map=&dummy,.map_len=1,.model_content_hash_ready=1,.snapshot_fingerprint_ready=1,
        .vision_numeric_hash_ready=1};
    /* Cached identities and no command list keep this entirely host-backed. */
    e.gpu.context=(ze_context_handle_t)(uintptr_t)1;
    xe_session s={.engine=&e};
    s.tokens=calloc(e.context,sizeof *s.tokens);s.logits=calloc(XE_VOCAB,sizeof *s.logits);
    s.swa_k=calloc(XE_SWA_SLAB_ELEMS,sizeof *s.swa_k);s.swa_v=calloc(XE_SWA_SLAB_ELEMS,sizeof *s.swa_v);
    s.global_k=calloc(xe_global_slab_elems(&e),sizeof *s.global_k);
    s.global_v=calloc(xe_global_slab_elems(&e),sizeof *s.global_v);
    CHECK(s.tokens&&s.logits&&s.swa_k&&s.swa_v&&s.global_k&&s.global_v);
    float *embeddings=calloc(70*XE_VISUAL_WIDTH,sizeof *embeddings);CHECK(embeddings);
    xe_visual_span span={.start=8,.rows=70,.budget=70,.embeddings=embeddings,.media=&media,.media_size=1};
    span.identity[0]=1;
    char path[]="/tmp/xenolith-visual-cache-XXXXXX";CHECK(mkdtemp(path));
    kvstore *store;CHECK(kvstore_open(&store,path,UINT64_C(1073741824))==KVSTORE_OK);
    attach(&s,&span,256);kvstore_id a=save(store,&s,256);
    uint64_t raw_a,raw_b,total,pinned;CHECK(xe_session_snapshot_size(&s,&raw_a)==XE_SNAPSHOT_OK);
    attach(&s,&span,512);kvstore_id b=save(store,&s,512);
    CHECK(xe_session_snapshot_size(&s,&raw_b)==XE_SNAPSHOT_OK);
    CHECK(kvstore_usage(store,&total,&pinned)==KVSTORE_OK&&total<raw_a+raw_b-4*1024*1024);
    uint64_t shared=total;
    CHECK(kvstore_committed(store,&b,&s,NULL,NULL)==KVSTORE_OK);
    CHECK(kvstore_contains(store,&a)==KVSTORE_OK); /* Compatible image reserve. */
    CHECK(kvstore_remove(store,&a)==KVSTORE_OK);
    kvstore_close(store);CHECK(kvstore_open(&store,path,UINT64_C(1073741824))==KVSTORE_OK);
    xe_input input={s.tokens,512,&span,1};
    CHECK(kvstore_load_input(store,&b,&s,&input,NULL)==KVSTORE_OK);
    fixture_state(&s,512,0,1);
    CHECK(s.visual_context&&s.visual_context->owned.input.visual[0].media[0]==media);
    span.identity[0]=2;
    xe_snapshot_status core;
    CHECK(kvstore_load_input(store,&b,&s,&input,&core)==KVSTORE_REJECTED&&core==XE_SNAPSHOT_TOKEN_MISMATCH);
    CHECK(s.n_tokens==512&&s.visual_context->owned.input.visual[0].identity[0]==1);
    span.identity[0]=1;embeddings[0]=1;
    CHECK(kvstore_load_input(store,&b,&s,&input,&core)==KVSTORE_REJECTED&&core==XE_SNAPSHOT_TOKEN_MISMATCH);
    embeddings[0]=0;media=2;
    CHECK(kvstore_load_input(store,&b,&s,&input,&core)==KVSTORE_REJECTED&&core==XE_SNAPSHOT_TOKEN_MISMATCH);
    media=1;
    span.budget=140;
    CHECK(kvstore_load_input(store,&b,&s,&input,&core)==KVSTORE_REJECTED&&core==XE_SNAPSHOT_TOKEN_MISMATCH);
    span.budget=70;
    /* V3 header, ABI and legacy visual-v2 recognition are independent of
     * upstream text-v2. Prefix checking never changes the live state. */
    FILE *f=tmpfile();CHECK(f&&xe_session_snapshot_save(&s,f)==XE_SNAPSHOT_OK);
    uint8_t h[384],original[384];CHECK(format_seek(f,0)&&format_read(f,h,384));memcpy(original,h,384);
    CHECK(format_get_u16le(h+8)==3&&format_get_u32le(h+24)==1&&format_get_u32le(h+352)==1);
    CHECK(xe_session_snapshot_check_prefix(&s,f,512)==XE_SNAPSHOT_OK);
    CHECK(xe_session_snapshot_check_prefix(&s,f,9)==XE_SNAPSHOT_TOKEN_MISMATCH);
    /* A later image must be omitted when checking an earlier checkpoint. */
    xe_visual_span pair[2]={span,span};pair[1].start=600;pair[1].identity[0]=3;
    fixture_state(&s,768,0,0);xe_input two={s.tokens,768,pair,2};
    xe_visual_context_release(s.visual_context);s.visual_context=xe_visual_context_new(&two);CHECK(s.visual_context);
    CHECK(xe_session_snapshot_check_prefix(&s,f,512)==XE_SNAPSHOT_OK);
    CHECK(xe_session_snapshot_check_prefix(&s,f,601)==XE_SNAPSHOT_TOKEN_MISMATCH);
    attach(&s,&span,512);
    format_put_u32le(h+352,2);rewrite_header(f,h);
    CHECK(xe_session_snapshot_load_input(&s,f,&input)==XE_SNAPSHOT_NUMERIC_MISMATCH&&s.n_tokens==512);
    memcpy(h,original,384);format_put_u16le(h+8,2);memset(h+352,0,24);rewrite_header(f,h);
    CHECK(xe_session_snapshot_load_input(&s,f,&input)==XE_SNAPSHOT_NUMERIC_MISMATCH&&s.n_tokens==512);
    memcpy(h,original,384);h[356]=1;rewrite_header(f,h);
    CHECK(xe_session_snapshot_load_input(&s,f,&input)==XE_SNAPSHOT_FORMAT);
    memcpy(h,original,384);format_put_u32le(h+24,0);rewrite_header(f,h);
    CHECK(xe_session_snapshot_load_input(&s,f,&input)==XE_SNAPSHOT_VERSION);
    rewrite_header(f,original);fclose(f);
    /* Same tokens/rows and same numerical KV, different image. The old
     * checkpoint must not be retained as a valid predecessor. */
    span.identity[0]=2;attach(&s,&span,768);kvstore_id c=save(store,&s,768);
    CHECK(kvstore_committed(store,&c,&s,NULL,NULL)==KVSTORE_OK);
    CHECK(kvstore_contains(store,&b)==KVSTORE_MISS&&kvstore_contains(store,&c)==KVSTORE_OK);
    CHECK(kvstore_usage(store,&total,&pinned)==KVSTORE_OK);uint64_t before=total;
    kvstore_close(store);CHECK(kvstore_open(&store,path,0)==KVSTORE_OK);
    input.count=768;CHECK(kvstore_load_input(store,&c,&s,&input,NULL)==KVSTORE_DISABLED);
    CHECK(kvstore_enforce_budget(store)==KVSTORE_DISABLED);
    kvstore_close(store);CHECK(kvstore_open(&store,path,UINT64_C(1073741824))==KVSTORE_OK);
    CHECK(kvstore_usage(store,&total,&pinned)==KVSTORE_OK&&total==before);
    CHECK(kvstore_load_input(store,&c,&s,&input,NULL)==KVSTORE_OK);fixture_state(&s,768,0,1);
    CHECK(kvstore_remove(store,&c)==KVSTORE_OK);
    CHECK(kvstore_usage(store,&total,&pinned)==KVSTORE_OK&&total==0);kvstore_close(store);
    char lock[256];snprintf(lock,sizeof lock,"%s/.lock",path);CHECK(unlink(lock)==0&&rmdir(path)==0);
    xe_visual_context_release(s.visual_context);free(embeddings);free(s.tokens);free(s.logits);
    free(s.swa_k);free(s.swa_v);free(s.global_k);free(s.global_v);
    printf("visual cache: v3/ABI, legacy-v2 rejection, image-safe reserves, dedup=%llu/%llu bytes, reopen and eviction PASS\n",
           (unsigned long long)shared,(unsigned long long)(raw_a+raw_b));return 0;
}
