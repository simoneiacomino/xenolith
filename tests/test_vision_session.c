#include "../xenolith.c"
#define CHECK(x) do { if(!(x)){fprintf(stderr,"vision session line %d: %s\n",__LINE__,#x);exit(1);} } while(0)

/* Uses a pinned projected fixture: no encoder run contaminates session tests. */
int main(int argc,char **argv) {
    if(argc!=4&&argc!=5){fprintf(stderr,"usage: test_vision_session MODEL PROMPT_IDS EMBEDDING [CONTEXT]\n");return 2;}
    int32_t ids[8192];int count=0;FILE *f=fopen(argv[2],"r");CHECK(f);
    while(count<8192&&fscanf(f,"%d",ids+count)==1){count++;int c=fgetc(f);if(c!=',')break;}
    fclose(f);f=fopen(argv[3],"rb");CHECK(f);CHECK(fseek(f,0,SEEK_END)==0);
    long bytes=ftell(f);CHECK(bytes>0&&bytes%(2816*4)==0);int rows=bytes/(2816*4);
    CHECK(rows>=1&&rows<=1120&&count>rows);rewind(f);
    float *embedding=malloc((size_t)bytes);CHECK(embedding);
    CHECK(fread(embedding,1,(size_t)bytes,f)==(size_t)bytes);CHECK(fgetc(f)==EOF);fclose(f);
    int context=argc==5?atoi(argv[4]):65536;CHECK(context==65536||context==131072);
    xe_engine *e=xe_engine_open_with_context(argv[1],context);
    int image=xe_token_id(e,"<|image|>"),start=0;while(start<count&&ids[start]!=image)start++;
    CHECK(start<count);xe_visual_span span={.start=start,.rows=rows,.embeddings=embedding};
    span.identity[0]=1;xe_input input={ids,count,&span,1};
    xe_session *s=xe_session_new(e);CHECK(xe_session_sync_input(s,&input));
    float *logits=malloc(XE_VOCAB*4);CHECK(logits);memcpy(logits,xe_session_logits(s),XE_VOCAB*4);
    CHECK(xe_session_sync_input(s,&input));CHECK(!memcmp(logits,xe_session_logits(s),XE_VOCAB*4));
    FILE *snapshot=tmpfile();CHECK(snapshot);uint64_t snapshot_bytes;
    CHECK(xe_session_snapshot_size(s,&snapshot_bytes)==XE_SNAPSHOT_OK);
    CHECK(xe_session_snapshot_save(s,snapshot)==XE_SNAPSHOT_OK);
    CHECK(xe_session_common_input(s,&input)==count);
    span.identity[0]=2;
    CHECK(xe_session_common_input(s,&input)==start);
    CHECK(xe_session_snapshot_load_input(s,snapshot,&input)==XE_SNAPSHOT_TOKEN_MISMATCH);
    CHECK(!memcmp(logits,xe_session_logits(s),XE_VOCAB*4));span.identity[0]=1;
    /* Same image ID with altered embedding bytes must also fail. */
    float saved=embedding[0];embedding[0]+=1;
    CHECK(xe_session_snapshot_load_input(s,snapshot,&input)==XE_SNAPSHOT_TOKEN_MISMATCH);embedding[0]=saved;
    xe_session_reset(s);CHECK(xe_session_snapshot_load_input(s,snapshot,&input)==XE_SNAPSHOT_OK);
    CHECK(!memcmp(logits,xe_session_logits(s),XE_VOCAB*4));
    CHECK(!xe_session_rewind_input(s,start+1));CHECK(s->n_tokens==count);
    ids[count]=9264;input.count=count+1;CHECK(xe_session_sync_input(s,&input));
    CHECK(xe_session_anchor_capture(s));
    xe_session *shadow=xe_session_shadow_new(s);CHECK(shadow);CHECK(shadow->visual_context==s->visual_context);
    void *borrowed_workspace=s->prefill_workspace;
    int borrowed_capacity=s->prefill_capacity;
    size_t borrowed_bytes=s->prefill_workspace_size;
    CHECK(shadow->prefill_workspace==borrowed_workspace);
    CHECK(shadow->prefill_capacity==borrowed_capacity&&shadow->prefill_workspace_size==borrowed_bytes);
    if(borrowed_capacity<XE_VISUAL_MAX) {
        CHECK(!xe_session_prefill_reserve(s,XE_VISUAL_MAX));
        CHECK(s->prefill_workspace==borrowed_workspace&&s->prefill_capacity==borrowed_capacity);
        CHECK(s->prefill_workspace_size==borrowed_bytes);
        CHECK(shadow->prefill_workspace==borrowed_workspace&&shadow->prefill_capacity==borrowed_capacity);
        CHECK(shadow->prefill_workspace_size==borrowed_bytes);
    }
    ids[count+1]=106;xe_tokens tail={ids,count+2,8192};CHECK(xe_session_shadow_sync(shadow,&tail)>=0);
    xe_session_shadow_refresh_logits(shadow);float *shadow_logits=malloc(XE_VOCAB*4);CHECK(shadow_logits);
    memcpy(shadow_logits,shadow->logits,XE_VOCAB*4);CHECK(xe_session_shadow_promote(s,shadow));xe_session_free(shadow);
    /* Promotion/free returns the SWA spares and ends the workspace borrow. */
    CHECK(xe_session_prefill_reserve(s,XE_VISUAL_MAX));
    CHECK(s->prefill_workspace&&s->prefill_capacity==XE_VISUAL_MAX);
    xe_prefill_workspace grown;
    CHECK(s->prefill_workspace_size==xe_prefill_workspace_layout_rows(&grown,NULL,XE_VISUAL_MAX));
    CHECK(s->n_tokens==count+2);CHECK(!memcmp(shadow_logits,s->logits,XE_VOCAB*4));
    /* Rewind/replay at an image boundary; memory belongs to the session. */
    xe_session_rewind(s,count);CHECK(s->n_tokens==count);
    CHECK(!memcmp(logits,xe_session_logits(s),XE_VOCAB*4));
    /* Append another complete image without discarding the previous KV, then
     * compare against a fresh replay of the same two-image input. */
    CHECK(count*2<=8192);memcpy(ids+count,ids,(size_t)count*4);
    xe_visual_span pair[2]={span,span};pair[1].start+=count;
    xe_input two={ids,count*2,pair,2};xe_sync_report sync;
    CHECK(xe_session_sync_input_report(s,&two,&sync));
    CHECK(sync.reused==count&&sync.prefilled==count&&!sync.restarted);
    xe_session *fresh=xe_session_new(e);CHECK(xe_session_sync_input(fresh,&two));
    CHECK(!memcmp(s->logits,fresh->logits,XE_VOCAB*4));xe_session_free(fresh);
    input.count=count;CHECK(xe_session_sync_input(s,&input));
    xe_tokens continuation={ids,count+1,8192};ids[count]=9264;
    CHECK(xe_session_sync_continuation(s,&continuation,NULL));
    ids[0]^=1;CHECK(!xe_session_sync_continuation(s,&continuation,NULL));ids[0]^=1;
    free(embedding);span.embeddings=NULL;
    xe_session_rewind(s,start+rows);CHECK(s->n_tokens==start+rows);
    xe_session_rewind(s,0);CHECK(s->n_tokens==0&&!s->visual_context&&!s->visual_prompt_end);
    free(shadow_logits);free(logits);fclose(snapshot);xe_session_free(s);xe_engine_close(e);
    puts("vision session: identity, snapshot, COW, promotion, replay and rewind PASS");return 0;
}
