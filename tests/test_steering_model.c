#include "../xenolith.c"
#include "steering_fixture.h"

int main(int argc,char **argv){
    if(argc!=2){fprintf(stderr,"usage: %s MODEL.gguf\n",argv[0]);return 2;}
    xe_engine *e=xe_engine_open_cached(argv[1],1024);
    for(int site=XE_STEERING_RESIDUAL;site<=XE_STEERING_FFN;site++){
    xe_session *a=xe_session_new(e),*b=xe_session_new(e);
    char root[]="/tmp/xe-steering-model-XXXXXX";assert(mkdtemp(root));char directory[4096];
    snprintf(directory,sizeof directory,"%s/artifact",root);
    xe_steering *v=steering_fixture_at(directory,XE_LAYERS,XE_EMBD,xe_engine_model_fingerprint(e),"projection",
        site==XE_STEERING_FFN?XE_STEERING_FFN_SITE:XE_STEERING_SITE);
    assert(xe_steering_get_site(v)==(xe_steering_site)site);
    assert(xe_session_steering_set(b,v,0));
    if(site==XE_STEERING_FFN){
        int32_t long_ids[601];for(int i=0;i<601;i++)long_ids[i]=2+i%101;
        xe_tokens long_prefix={long_ids,600,601};
        xe_session *capture=xe_session_new(e);assert(xe_session_capture_enable_at(capture,XE_STEERING_FFN));
        assert(xe_session_set_text_prefill(a,1024));assert(xe_session_set_text_prefill(b,1024));
        assert(xe_session_set_text_prefill(capture,1024));
        xe_session_sync(a,&long_prefix);xe_session_sync(b,&long_prefix);xe_session_sync(capture,&long_prefix);
        assert(!memcmp(a->hidden,b->hidden,XE_EMBD*sizeof(float)));
        assert(!memcmp(a->logits,b->logits,XE_VOCAB*sizeof(float)));
        assert(!memcmp(a->hidden,capture->hidden,XE_EMBD*sizeof(float)));
        assert(!memcmp(a->logits,capture->logits,XE_VOCAB*sizeof(float)));
        xe_session_free(capture);xe_session_reset(a);xe_session_reset(b);
        assert(xe_session_steering_set(a,v,0.2f));
        xe_session_sync(a,&long_prefix);xe_session_sync(b,&long_prefix);
        assert(memcmp(a->logits,b->logits,XE_VOCAB*sizeof(float)));
        for(int i=0;i<XE_VOCAB;i++)assert(isfinite(a->logits[i]));
        FILE *large=tmpfile();assert(large);assert(xe_session_snapshot_save(a,large)==XE_SNAPSHOT_OK);
        assert(xe_session_steering_schedule(b,a->steering_steps,a->steering_count));
        assert(xe_session_snapshot_load(b,large,&long_prefix)==XE_SNAPSHOT_OK);fclose(large);
        assert(!memcmp(a->logits,b->logits,XE_VOCAB*sizeof(float)));
        long_prefix.len=601;xe_session_sync(a,&long_prefix);xe_session_sync(b,&long_prefix);
        assert(!memcmp(a->hidden,b->hidden,XE_EMBD*sizeof(float)));
        assert(!memcmp(a->logits,b->logits,XE_VOCAB*sizeof(float)));
        xe_session_reset(a);xe_session_reset(b);
        assert(xe_session_steering_schedule(a,NULL,0));assert(xe_session_steering_schedule(b,NULL,0));
        assert(xe_session_steering_set(b,v,0));
    }
    int32_t ids[12];for(int i=0;i<12;i++)ids[i]=i+2;
    xe_tokens prefix={ids,8,12};xe_session_sync(a,&prefix);xe_session_sync(b,&prefix);
    assert(!memcmp(a->logits,b->logits,XE_VOCAB*sizeof(float)));
    assert(!memcmp(a->hidden,b->hidden,XE_EMBD*sizeof(float)));
    if(site==XE_STEERING_FFN){
        xe_session *capture=xe_session_new(e);assert(xe_session_capture_enable_at(capture,XE_STEERING_FFN));
        xe_session_sync(capture,&prefix);
        assert(!memcmp(a->hidden,capture->hidden,XE_EMBD*sizeof(float)));
        assert(!memcmp(a->logits,capture->logits,XE_VOCAB*sizeof(float)));
        assert(memcmp(capture->capture_rows+(XE_LAYERS-1)*XE_EMBD,capture->hidden,XE_EMBD*sizeof(float)));
        xe_session_free(capture);
    }
    prefix.len=9;xe_session_sync(a,&prefix);xe_session_sync(b,&prefix);
    assert(!memcmp(a->logits,b->logits,XE_VOCAB*sizeof(float)));
    assert(!memcmp(a->hidden,b->hidden,XE_EMBD*sizeof(float)));
    float *logits=malloc(XE_VOCAB*sizeof *logits);assert(logits);memcpy(logits,a->logits,XE_VOCAB*sizeof *logits);
    FILE *before=tmpfile(),*pending=tmpfile();assert(before&&pending);
    assert(xe_session_snapshot_save(a,before)==XE_SNAPSHOT_OK);
    assert(xe_session_steering_set(a,v,0.5f));
    assert(a->n_tokens==9&&!memcmp(logits,a->logits,XE_VOCAB*sizeof *logits));
    assert(xe_session_snapshot_save(a,pending)==XE_SNAPSHOT_OK);
    rewind(before);rewind(pending);
    int x,y;do{x=fgetc(before);y=fgetc(pending);assert(x==y);}while(x!=EOF);
    fclose(before);fclose(pending);
    prefix.len=10;xe_session_sync(a,&prefix);xe_session_sync(b,&prefix);
    assert(memcmp(a->logits,b->logits,XE_VOCAB*sizeof(float)));
    FILE *snapshot=tmpfile();assert(snapshot);assert(xe_session_snapshot_save(a,snapshot)==XE_SNAPSHOT_OK);
    memcpy(logits,b->logits,XE_VOCAB*sizeof *logits);int old=b->n_tokens;
    assert(xe_session_snapshot_load(b,snapshot,&prefix)==XE_SNAPSHOT_NUMERIC_MISMATCH);
    assert(b->n_tokens==old&&!memcmp(logits,b->logits,XE_VOCAB*sizeof *logits));
    assert(xe_session_steering_schedule(b,a->steering_steps,a->steering_count));
    assert(xe_session_snapshot_load(b,snapshot,&prefix)==XE_SNAPSHOT_OK);
    assert(!memcmp(a->logits,b->logits,XE_VOCAB*sizeof(float)));
    prefix.len=11;xe_session_sync(a,&prefix);xe_session_sync(b,&prefix);
    assert(!memcmp(a->logits,b->logits,XE_VOCAB*sizeof(float)));
    assert(!memcmp(a->hidden,b->hidden,XE_EMBD*sizeof(float)));
    fclose(snapshot);free(logits);xe_session_free(a);xe_session_free(b);
    xe_steering_release(v);steering_fixture_remove(directory);rmdir(root);
    }
    xe_engine_close(e);
    puts("steering model (residual and FFN sites): byte-exact zero strength, cached logits, pending snapshot identity, schedule mismatch and restored continuation PASS");
}
