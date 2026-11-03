#ifndef TEST_STEERING_FIXTURE_H
#define TEST_STEERING_FIXTURE_H
#include "../json.h"
#include "../format.h"
#include "../steering.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static xe_steering *steering_fixture_at(const char *directory,int layers,int width,
                                    const uint8_t model[32],const char *mode,const char *site) {
    assert(mkdir(directory,0700)==0);
    size_t n=(size_t)layers*width;uint8_t *payload=calloc(n,4);assert(payload);
    for(int l=0;l<layers;l++) {
        if(l!=0&&l!=layers-1)continue;
        double norm=0;
        for(int j=0;j<width;j++){double x=sin((j+1)*(l+1)*0.01);norm+=x*x;}
        for(int j=0;j<width;j++){
            float value=(float)(sin((j+1)*(l+1)*0.01)/sqrt(norm));uint32_t bits;memcpy(&bits,&value,4);
            format_put_u32le(payload+4*((size_t)l*width+j),bits);
        }
    }
    uint8_t hash[32];char mh[65],ph[65],path[4096];
    xe_steering_hex(model,mh);format_sha256_bytes(payload,n*4,hash);xe_steering_hex(hash,ph);
    snprintf(path,sizeof path,"%s/vectors.f32",directory);FILE *f=fopen(path,"wb");assert(f);
    assert(fwrite(payload,4,n,f)==n);assert(!fclose(f));free(payload);
    const char *zero="0000000000000000000000000000000000000000000000000000000000000000";
    snprintf(path,sizeof path,"%s/manifest.json",directory);f=fopen(path,"wb");assert(f);
    fprintf(f,"{\"version\":1,\"layers\":%d,\"width\":%d,\"dtype\":\"f32le\",\"site\":\"%s\","
        "\"mode\":\"%s\",\"normalization\":\"%s\",\"payload\":\"vectors.f32\",\"payload_sha256\":\"%s\","
        "\"model_sha256\":\"%s\",\"active_layers\":[0,%d],\"token_policy\":\"all_text\","
        "\"provenance\":{\"model_source\":\"synthetic fixture\",\"model_revision\":\"fixture\",\"dataset_sha256\":\"%s\","
        "\"template_sha256\":\"%s\",\"method\":\"synthetic\",\"position\":\"last\",\"reasoning\":\"off\","
        "\"engine_commit\":\"fixture\",\"numeric_settings\":\"fixture\",\"positive_label\":\"a\","
        "\"negative_label\":\"b\",\"evaluation\":\"test only\"}}\n",layers,width,site,mode,!strcmp(mode,"projection")?"unit":"none",ph,mh,layers-1,zero,zero);
    assert(!fclose(f));char error[256];xe_steering *v=NULL;
    assert(xe_steering_load(path,layers,width,model,&v,error,sizeof error));return v;
}
static xe_steering *steering_fixture(const char *directory,int layers,int width,
                                    const uint8_t model[32],const char *mode) {
    return steering_fixture_at(directory,layers,width,model,mode,XE_STEERING_SITE);
}
static void steering_fixture_remove(const char *directory){
    char path[4096];snprintf(path,sizeof path,"%s/vectors.f32",directory);unlink(path);
    snprintf(path,sizeof path,"%s/manifest.json",directory);unlink(path);rmdir(directory);
}
#endif
