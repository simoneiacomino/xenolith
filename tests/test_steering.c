#define _POSIX_C_SOURCE 200809L
#define _DARWIN_C_SOURCE
#include "../steering.h"
#include "../format.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static void fixture(const char *dir, const char *mode, int corrupt) {
    char path[512], payload_hash[65]; uint8_t bytes[24]={0}, hash[32];
    float values[6]={0.6f,0.8f,0,0,0,0};
    if(corrupt==2)values[0]=NAN;
    if(corrupt==3)values[0]=2;
    if(corrupt==4)values[3]=1;
    if(corrupt==5)values[0]=values[1]=0;
    for(int i=0;i<6;i++){uint32_t bits;memcpy(&bits,values+i,4);format_put_u32le(bytes+4*i,bits);}
    format_sha256_bytes(bytes,sizeof bytes,hash);xe_steering_hex(hash,payload_hash);
    snprintf(path,sizeof path,"%s/vectors.f32",dir);FILE *f=fopen(path,"wb");assert(f);
    if(corrupt==1)bytes[3]^=1;
    assert(fwrite(bytes,1,sizeof bytes,f)==sizeof bytes);assert(!fclose(f));
    snprintf(path,sizeof path,"%s/manifest.json",dir);f=fopen(path,"wb");assert(f);
    const char *zero="0000000000000000000000000000000000000000000000000000000000000000";
    const char *site=corrupt==6?XE_STEERING_FFN_SITE:corrupt==7?"unsupported_site":XE_STEERING_SITE;
    fprintf(f,"{\"version\":1,\"layers\":2,\"width\":3,\"dtype\":\"f32le\",\"site\":\"%s\","
        "\"mode\":\"%s\",\"normalization\":\"%s\",\"payload\":\"vectors.f32\",\"payload_sha256\":\"%s\","
        "\"model_sha256\":\"%s\",\"active_layers\":[0],\"token_policy\":\"all_text\","
        "\"provenance\":{\"model_source\":\"fixture\",\"model_revision\":\"fixture\",\"dataset_sha256\":\"%s\","
        "\"template_sha256\":\"%s\",\"method\":\"test\",\"position\":\"last\",\"reasoning\":\"off\","
        "\"engine_commit\":\"fixture\",\"numeric_settings\":\"fixture\",\"positive_label\":\"a\","
        "\"negative_label\":\"b\",\"evaluation\":\"test only\"}}\n",site,mode,!strcmp(mode,"projection")?"unit":"none",payload_hash,zero,zero,zero);
    assert(!fclose(f));
}
int main(void){
    float h[3]={3,4,7},v[3]={0.6f,0.8f,0};
    xe_steering_apply(h,v,3,XE_STEERING_PROJECTION,1);
    assert(fabsf(h[0])<1e-6f&&fabsf(h[1])<1e-6f&&h[2]==7);
    float a[3]={3,4,7};xe_steering_apply(a,v,3,XE_STEERING_PROJECTION,-1);
    assert(fabsf(a[0]-6)<1e-6f&&fabsf(a[1]-8)<1e-6f&&a[2]==7);
    float b[3]={3,4,7},before[3];memcpy(before,b,sizeof b);
    xe_steering_apply(b,v,3,XE_STEERING_ADDITIVE,0);assert(!memcmp(b,before,sizeof b));
    xe_steering_apply(b,v,3,XE_STEERING_ADDITIVE,2);assert(fabsf(b[0]-4.2f)<1e-6f);
    char root[]="/tmp/xe-steering-unit-XXXXXX";assert(mkdtemp(root));
    char artifact[512],store[512],path[512],error[256],id[65];
    snprintf(artifact,sizeof artifact,"%s/input",root);assert(!mkdir(artifact,0700));
    snprintf(store,sizeof store,"%s/store",root);assert(!mkdir(store,0700));
    snprintf(path,sizeof path,"%s/input/manifest.json",root);
    uint8_t model[32]={0};xe_steering *vector=NULL;
    fixture(artifact,"projection",0);
    assert(xe_steering_load(path,2,3,model,&vector,error,sizeof error));
    assert(xe_steering_get_site(vector)==XE_STEERING_RESIDUAL);
    assert(xe_steering_active(vector,0)&&!xe_steering_active(vector,1));
    assert(xe_steering_row(vector,1)==NULL);
    assert(xe_steering_import(store,path,2,3,model,id,error,sizeof error));
    assert(!strcmp(id,xe_steering_id(vector)));
    assert(xe_steering_import(store,path,2,3,model,id,error,sizeof error));
    xe_steering *copy=NULL;assert(xe_steering_open_id(store,id,2,3,model,&copy,error,sizeof error));
    assert(!strcmp(xe_steering_id(copy),id));xe_steering_release(copy);
    xe_steering *unchanged=vector;
    model[0]=1;assert(!xe_steering_load(path,2,3,model,&unchanged,error,sizeof error));assert(unchanged==vector);model[0]=0;
    assert(!xe_steering_load(path,3,3,model,&unchanged,error,sizeof error));assert(unchanged==vector);
    fixture(artifact,"projection",1);assert(!xe_steering_load(path,2,3,model,&unchanged,error,sizeof error));assert(unchanged==vector);
    for(int invalid=2;invalid<=5;invalid++) {
        fixture(artifact,"projection",invalid);
        assert(!xe_steering_load(path,2,3,model,&unchanged,error,sizeof error));assert(unchanged==vector);
    }
    fixture(artifact,"projection",6);
    assert(xe_steering_load(path,2,3,model,&copy,error,sizeof error));
    assert(xe_steering_get_site(copy)==XE_STEERING_FFN && strcmp(xe_steering_id(copy),id));xe_steering_release(copy);
    fixture(artifact,"projection",7);assert(!xe_steering_load(path,2,3,model,&unchanged,error,sizeof error));
    fixture(artifact,"projection",0);
    char payload_path[512];snprintf(payload_path,sizeof payload_path,"%s/input/vectors.f32",root);
    FILE *extra=fopen(payload_path,"ab");assert(extra);fputc(0,extra);assert(!fclose(extra));
    assert(!xe_steering_load(path,2,3,model,&unchanged,error,sizeof error));assert(unchanged==vector);
    fixture(artifact,"projection",0);assert(!truncate(payload_path,4));
    assert(!xe_steering_load(path,2,3,model,&unchanged,error,sizeof error));assert(unchanged==vector);
    fixture(artifact,"projection",0);
    FILE *duplicate=fopen(path,"r+");assert(duplicate);assert(!fseek(duplicate,-2,SEEK_END));
    assert(fputs(",\"version\":1}\n",duplicate)>=0);assert(!fclose(duplicate));
    assert(!xe_steering_load(path,2,3,model,&unchanged,error,sizeof error));assert(unchanged==vector);
    fixture(artifact,"unsupported",0);assert(!xe_steering_load(path,2,3,model,&unchanged,error,sizeof error));
    assert(!xe_steering_open_id(store,"../bad",2,3,model,&unchanged,error,sizeof error));
    fixture(artifact,"projection",0);
    char renamed[512];snprintf(renamed,sizeof renamed,"%s/input/renamed.json",root);
    assert(!rename(path,renamed));
    assert(xe_steering_load(renamed,2,3,model,&copy,error,sizeof error));
    assert(!strcmp(id,xe_steering_id(copy)));xe_steering_release(copy);assert(!rename(renamed,path));
    char stored_payload[1024];snprintf(stored_payload,sizeof stored_payload,"%s/%s/vectors.f32",store,id);
    extra=fopen(stored_payload,"ab");assert(extra);fputc(0,extra);assert(!fclose(extra));
    assert(!xe_steering_import(store,path,2,3,model,id,error,sizeof error));
    xe_steering_release(vector);
    char target[1024];snprintf(target,sizeof target,"%s/%s/manifest.json",store,id);unlink(target);
    snprintf(target,sizeof target,"%s/%s/vectors.f32",store,id);unlink(target);
    snprintf(target,sizeof target,"%s/%s",store,id);rmdir(target);rmdir(store);
    snprintf(target,sizeof target,"%s/vectors.f32",artifact);unlink(target);unlink(path);rmdir(artifact);rmdir(root);
    puts("steering loader, identity, import, failure atomicity and analytical operations: PASS");
}
