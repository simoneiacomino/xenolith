/* Real media parsing, persistence and embedding cache, with a deterministic
 * encoder so weight changes and replay can be checked without a GPU/model. */
#include "../xenolith.c"
#include "../profile.h"
#define CHECK(x) do { if(!(x)){fprintf(stderr,"visual replay line %d: %s\n",__LINE__,#x);exit(1);} } while(0)

static int encodes,decodes,image_frees;
static int encode_ok=1;
static const xe_image *expected_image;
static int fixture_encode(xe_engine *e,const xe_image *im,float *out,const char *dump) {
    (void)dump;encodes++;
    CHECK(!expected_image||im==expected_image);
    for(size_t i=0;i<(size_t)im->rows*XE_VISUAL_WIDTH;i++)out[i]=e->vision_model.sha256[0];
    return encode_ok;
}
static int fixture_image_load(xe_image *im,const unsigned char *bytes,size_t size,
                              int budget,char *error,size_t cap) {
    decodes++;
    return xe_image_load_memory(im,bytes,size,budget,error,cap);
}
static void fixture_image_free(xe_image *im) {
    image_frees++;
    xe_image_free(im);
}
/* Keep insertion on the real preprocessing/storage path without a tokenizer. */
static profile_status fixture_render_image(profile *p,const char *text,uint32_t rows,
                                           uint32_t turn,profile_render *out,uint32_t *offset) {
    (void)p;(void)text;(void)turn;
    static int32_t tokens[XE_VISUAL_MAX];
    CHECK(rows<=XE_VISUAL_MAX);
    *out=(profile_render){.render=(const uint8_t *)"",.tokens=tokens,.token_count=rows};
    *offset=0;return PROFILE_OK;
}
#define xe_engine_encode_image fixture_encode
#define xe_image_load_memory fixture_image_load
#define xe_image_free fixture_image_free
#define profile_render_user_image fixture_render_image
#undef _POSIX_C_SOURCE
#include "../runtime.c"
#undef xe_engine_encode_image
#undef xe_image_load_memory
#undef xe_image_free
#undef profile_render_user_image

int main(void) {
    static const uint8_t png[] = {137,80,78,71,13,10,26,10,0,0,0,13,73,72,68,82,0,0,0,1,0,0,0,1,8,2,0,0,0,144,119,83,222,0,0,0,12,73,68,65,84,120,156,99,248,207,192,0,0,3,1,1,0,201,254,146,239,0,0,0,0,73,69,78,68,174,66,96,130};
    xe_engine e={.context=65536,.pool_initialized=1,.owner=pthread_self(),
                 .model_content_hash_ready=1,.vision_weights=(void *)(uintptr_t)1};
    e.model_content_hash[0]=1;e.vision_model.sha256[0]=1;
    runtime w={.engine=&e,.context=65536,.vision_numeric_ready=1};
    w.vision_numeric[0]=1;
    const char *tmp=getenv("TMPDIR");if(!tmp)tmp="/tmp";
    char directory[4096];CHECK(snprintf(directory,sizeof directory,"%s/xenolith-replay-XXXXXX",tmp)<(int)sizeof directory);
    CHECK(mkdtemp(directory));
    CHECK(snprintf(w.vision_cache_dir,sizeof w.vision_cache_dir,"%s/embeddings",directory)<(int)sizeof w.vision_cache_dir);
    xe_image im;char error[256];CHECK(fixture_image_load(&im,png,sizeof png,70,error,sizeof error));
    xe_image saved=im;
    runtime_visual_begin(&w);
    runtime_visual_asset *asset;
    expected_image=&im;
    CHECK(runtime_visual_asset_get(&w,&im,png,sizeof png,NULL,&im,&asset)==RUNTIME_OK && encodes==1);
    CHECK(decodes==1&&image_frees==0&&!memcmp(&im,&saved,sizeof im));
    CHECK(!memcmp(im.encoded,png,sizeof png)&&im.rgb[0]==im.resized[0]);
    CHECK(asset->media!=im.encoded&&!memcmp(asset->media,png,sizeof png));
    expected_image=NULL;
    uint8_t record[XE_IMAGE_RECORD_HEADER+sizeof png]={0};
    memcpy(record,"XEVIMG1\n",8);
    format_put_u32le(record+12,im.rows);format_put_u32le(record+16,im.budget);
    format_put_u32le(record+20,im.width);format_put_u32le(record+24,im.height);
    format_put_u32le(record+28,im.resized_width);format_put_u32le(record+32,im.resized_height);
    format_put_u32le(record+36,XE_IMAGE_RECORD_HEADER);format_put_u64le(record+40,sizeof png);
    memcpy(record+48,im.encoded_hash,32);memcpy(record+80,im.rgb_hash,32);
    memcpy(record+112,asset->identity,32);memcpy(record+144,e.vision_model.sha256,32);
    memcpy(record+176,e.model_content_hash,32);memcpy(record+208,w.vision_numeric,32);
    memcpy(record+240,im.resized_hash,32);memcpy(record+XE_IMAGE_RECORD_HEADER,png,sizeof png);
    conversation_store *store;conversation *c;conversation_id id;
    CHECK(conversation_store_open(&store,directory)==CONVERSATION_OK);
    CHECK(conversation_create(store,&c,&id)==CONVERSATION_OK);
    conversation_block block={CONVERSATION_BLOCK_IMAGE,record,sizeof record};
    int32_t tokens[1120]={0};
    CHECK(conversation_append_message(c,CONVERSATION_ROLE_USER,&block,1,"",0,tokens,im.rows)==CONVERSATION_OK);
    CHECK(conversation_commit(c)==CONVERSATION_OK);
    conversation_close(c);
    CHECK(conversation_open(store,&id,&c)==CONVERSATION_OK);
    CHECK(conversation_project(c,0,0)==CONVERSATION_OK);
    runtime_visual_release(&w);
    CHECK(runtime_visual_collect(&w,c)==RUNTIME_OK && encodes==1); /* Disk reuse. */
    CHECK(decodes==1&&image_frees==0);
    uint8_t original[32];memcpy(original,w.visual[0].identity,32);
    e.model_content_hash[0]=2;
    CHECK(runtime_visual_collect(&w,c)==RUNTIME_OK && encodes==2);
    CHECK(decodes==2&&image_frees==1); /* Replay owns and frees its preprocessing. */
    CHECK(memcmp(original,w.visual[0].identity,32));
    CHECK(runtime_visual_collect(&w,c)==RUNTIME_OK && encodes==2); /* Memory reuse. */
    CHECK(decodes==2&&image_frees==1);
    e.model_content_hash[0]=1;
    CHECK(runtime_visual_collect(&w,c)==RUNTIME_OK && encodes==2); /* Original cache remains valid. */
    CHECK(!memcmp(original,w.visual[0].identity,32));
    e.vision_model.sha256[0]=2;
    CHECK(runtime_visual_collect(&w,c)==RUNTIME_OK && encodes==3);
    CHECK(w.visual[0].embeddings[0]==2 && memcmp(original,w.visual[0].identity,32));
    w.vision_numeric[0]=2;
    CHECK(runtime_visual_collect(&w,c)==RUNTIME_OK && encodes==4);
    /* Old metadata identities permit media replay, never an old-rows assertion. */
    uint8_t legacy[sizeof record];memcpy(legacy,record,sizeof record);
    memset(legacy+144,0x55,64);memcpy(legacy+208,w.vision_numeric,32);legacy[112]^=1;
    CHECK(runtime_visual_asset_get(&w,&im,png,sizeof png,legacy,NULL,&asset)==RUNTIME_OK);
    /* Same weights/numerics must still detect inconsistent recorded rows. */
    memcpy(legacy+144,e.vision_model.sha256,32);memcpy(legacy+176,e.model_content_hash,32);
    CHECK(runtime_visual_asset_get(&w,&im,png,sizeof png,legacy,NULL,&asset)==RUNTIME_INVALID_ARGUMENT);
    uint8_t corrupt[sizeof png];memcpy(corrupt,png,sizeof png);corrupt[sizeof png-1]^=1;
    CHECK(runtime_visual_asset_get(&w,&im,corrupt,sizeof corrupt,NULL,NULL,&asset)==RUNTIME_INVALID_ARGUMENT);
    CHECK(w.visual_count==1 && !memcmp(w.visual[0].media,png,sizeof png));

    runtime scratch={.engine=&e,.context=65536,.vision_numeric_ready=1};
    int before_decodes=decodes,before_frees=image_frees,before_encodes=encodes;
    xe_image bad=im;bad.rgb_hash[0]^=1;
    CHECK(runtime_visual_asset_get(&scratch,&bad,png,sizeof png,NULL,&im,&asset)==RUNTIME_INVALID_ARGUMENT);
    CHECK(decodes==before_decodes&&image_frees==before_frees&&encodes==before_encodes);
    encode_ok=0;expected_image=&im;
    CHECK(runtime_visual_asset_get(&scratch,&im,png,sizeof png,NULL,&im,&asset)==RUNTIME_INVALID_ARGUMENT);
    CHECK(decodes==before_decodes&&image_frees==before_frees&&encodes==before_encodes+1);
    CHECK(!memcmp(&im,&saved,sizeof im)&&!memcmp(im.encoded,png,sizeof png));
    CHECK(im.rgb[0]==im.resized[0]); /* Borrowed pixels survive failed encoding too. */
    encode_ok=1;expected_image=NULL;
    CHECK(runtime_visual_asset_get(&scratch,&bad,png,sizeof png,NULL,NULL,&asset)==RUNTIME_INVALID_ARGUMENT);
    CHECK(decodes==before_decodes+1&&image_frees==before_frees+1&&encodes==before_encodes+1);
    CHECK(!scratch.vision_asset_count); /* Replay still rejects changed preprocessing. */

    before_decodes=decodes;before_frees=image_frees;before_encodes=encodes;
    runtime_message message={.kind=RUNTIME_MESSAGE_USER,.image=png,.image_length=sizeof png,.image_budget=70};
    profile_render render;uint8_t *payload;size_t payload_size;uint32_t offset;
    CHECK(runtime_visual_prepare(&scratch,&message,NULL,PROFILE_TURN_PADDED,0,
          &render,&payload,&payload_size,&asset,&offset)==RUNTIME_OK);
    CHECK(decodes==before_decodes+1&&image_frees==before_frees+1&&encodes==before_encodes+1);
    CHECK(payload_size==sizeof record&&offset==0);
    CHECK(!memcmp(payload+48,record+48,64)&&!memcmp(payload+240,record+240,32));
    CHECK(!memcmp(payload+XE_IMAGE_RECORD_HEADER,png,sizeof png));
    CHECK(asset->rows==im.rows&&asset->embeddings[0]==e.vision_model.sha256[0]);
    CHECK(!memcmp(asset->media,png,sizeof png)); /* Asset owns bytes after prepared image release. */
    free(payload);runtime_visual_release(&scratch);
    runtime_visual_release(&w);xe_image_free(&im);
    conversation_close(c);CHECK(conversation_delete(store,&id)==CONVERSATION_OK);
    conversation_store_close(store);
    puts("visual replay: prepared image borrowing, single preprocessing, replay ownership and persistence PASS");
    return 0;
}
