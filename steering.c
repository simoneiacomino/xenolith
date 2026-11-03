#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#include "steering.h"
#include "format.h"
#include "json.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

struct xe_steering {
    size_t refs;
    int layers, width;
    xe_steering_mode mode;
    xe_steering_site site;
    uint8_t model[32];
    char id[65];
    uint8_t *mask;
    float *rows;
    char *manifest;
    size_t manifest_size;
    uint8_t *payload;
    size_t payload_size;
};

static int fail(char *error, size_t cap, const char *s) {
    if (error && cap) snprintf(error, cap, "%s", s);
    return 0;
}
void xe_steering_hex(const uint8_t b[32], char out[65]) {
    static const char h[]="0123456789abcdef";
    for (int i=0;i<32;i++) {out[2*i]=h[b[i]>>4];out[2*i+1]=h[b[i]&15];} out[64]=0;
}
int xe_steering_valid_id(const char *s) {
    if (!s || strlen(s)!=64) return 0;
    for (int i=0;i<64;i++) if (!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f'))) return 0;
    return 1;
}
int xe_steering_unhex(const char *s, uint8_t out[32]) {
    if (!xe_steering_valid_id(s)) return 0;
    for (int i=0;i<32;i++) {
        unsigned a=(unsigned)(s[2*i]<='9'?s[2*i]-'0':s[2*i]-'a'+10);
        unsigned b=(unsigned)(s[2*i+1]<='9'?s[2*i+1]-'0':s[2*i+1]-'a'+10);
        out[i]=(uint8_t)((a<<4)|b);
    } return 1;
}
static uint8_t *read_bounded(const char *path, size_t max, size_t *n) {
    int fd=open(path,O_RDONLY|O_CLOEXEC|O_NOFOLLOW);
    struct stat st;
    if(fd<0)return NULL;
    if(fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0||(uint64_t)st.st_size>max){close(fd);return NULL;}
    size_t size=(size_t)st.st_size;
    uint8_t *p=malloc(size+1); size_t at=0;
    if(!p){close(fd);return NULL;}
    while(at<size){ssize_t r=read(fd,p+at,size-at);if(r<0&&errno==EINTR)continue;if(r<=0)break;at+=(size_t)r;}
    uint8_t extra;ssize_t tail=read(fd,&extra,1);close(fd);
    if(at!=size||tail!=0){free(p);return NULL;}
    p[size]=0;*n=size;return p;
}
static int string_is(const json_value *j,const char *s){return j&&j->type==JSON_STRING&&!strcmp(j->text,s);}
static int unique_members(const json_value *j) {
    if (!j) return 0;
    for (const json_value *a=j->child;a;a=a->next) {
        if (j->type==JSON_OBJECT) for (const json_value *b=a->next;b;b=b->next)
            if (!strcmp(a->key,b->key)) return 0;
        if ((a->type==JSON_OBJECT||a->type==JSON_ARRAY)&&!unique_members(a)) return 0;
    }
    return 1;
}
static int number_is(const json_value *j,int n){return j&&j->type==JSON_NUMBER&&j->number==n;}
static const char *str(const json_value *j,const char *key){const json_value *v=json_member(j,key);return v&&v->type==JSON_STRING?v->text:NULL;}

int xe_steering_load(const char *path,int layers,int width,const uint8_t model[32],
                     xe_steering **out,char *error,size_t cap) {
    if(!path||!out||!model||layers<1||layers>256||width<1||width>65536)
        return fail(error,cap,"invalid steering arguments");
    xe_steering *v=calloc(1,sizeof *v);json_value *j=NULL;
    if(!v)return fail(error,cap,"out of memory loading steering");
    v->refs=1;v->layers=layers;v->width=width;
    v->manifest=(char *)read_bounded(path,65536,&v->manifest_size);
    const char *message="cannot read steering manifest";
    if(!v->manifest)goto bad;
    j=json_parse(v->manifest,v->manifest_size);
    message="incompatible steering manifest";
    if(!j||j->type!=JSON_OBJECT||!unique_members(j)||!number_is(json_member(j,"version"),1)||
       !number_is(json_member(j,"layers"),layers)||!number_is(json_member(j,"width"),width)||
       !string_is(json_member(j,"dtype"),"f32le")||
       !string_is(json_member(j,"payload"),"vectors.f32")||
       !string_is(json_member(j,"token_policy"),"all_text"))goto bad;
    if(string_is(json_member(j,"site"),XE_STEERING_SITE))v->site=XE_STEERING_RESIDUAL;
    else if(string_is(json_member(j,"site"),XE_STEERING_FFN_SITE))v->site=XE_STEERING_FFN;
    else goto bad;
    if(string_is(json_member(j,"mode"),"projection"))v->mode=XE_STEERING_PROJECTION;
    else if(string_is(json_member(j,"mode"),"additive"))v->mode=XE_STEERING_ADDITIVE;
    else goto bad;
    if(!string_is(json_member(j,"normalization"),v->mode==XE_STEERING_PROJECTION?"unit":"none"))goto bad;
    message="steering model fingerprint mismatch";
    if(!xe_steering_unhex(str(j,"model_sha256"),v->model)||memcmp(model,v->model,32))goto bad;
    message="missing steering provenance";
    const json_value *provenance=json_member(j,"provenance");
    const char *required[]={"model_source","model_revision","dataset_sha256","template_sha256",
        "method","position","reasoning","engine_commit","numeric_settings","positive_label","negative_label","evaluation"};
    if(!provenance||provenance->type!=JSON_OBJECT)goto bad;
    for(size_t i=0;i<sizeof required/sizeof *required;i++)if(!str(provenance,required[i])||!*str(provenance,required[i]))goto bad;
    uint8_t hash[32],expected[32];
    if(!xe_steering_unhex(str(provenance,"dataset_sha256"),hash)||
       !xe_steering_unhex(str(provenance,"template_sha256"),hash))goto bad;
    message="invalid steering layer mask";
    const json_value *mask=json_member(j,"active_layers");
    if(!mask||mask->type!=JSON_ARRAY||!json_length(mask)||json_length(mask)>(size_t)layers)goto bad;
    v->mask=calloc((size_t)layers,1);
    if(!v->mask)goto nomem;
    for(const json_value *p=mask->child;p;p=p->next){
        if(p->type!=JSON_NUMBER||!isfinite(p->number)||p->number<0||p->number>=layers||floor(p->number)!=p->number)goto bad;
        int l=(int)p->number;if(v->mask[l])goto bad;v->mask[l]=1;
    }
    size_t bytes=(size_t)layers*width*4;
    char payload_path[4096]; const char *slash=strrchr(path,'/');
    int count=slash?snprintf(payload_path,sizeof payload_path,"%.*s/vectors.f32",(int)(slash-path),path)
                   :snprintf(payload_path,sizeof payload_path,"vectors.f32");
    message="cannot read steering payload";
    if(count<0||(size_t)count>=sizeof payload_path)goto bad;
    v->payload=read_bounded(payload_path,bytes,&v->payload_size);
    if(!v->payload||v->payload_size!=bytes)goto bad;
    message="steering payload checksum mismatch";
    format_sha256_bytes(v->payload,bytes,hash);
    if(!xe_steering_unhex(str(j,"payload_sha256"),expected)||memcmp(hash,expected,32))goto bad;
    v->rows=malloc(bytes);if(!v->rows)goto nomem;
    message="invalid steering vector values or normalization";
    for(int l=0;l<layers;l++){
        double norm=0;
        for(int k=0;k<width;k++){
            size_t i=(size_t)l*width+k;uint32_t bits=format_get_u32le(v->payload+4*i);
            float f;memcpy(&f,&bits,4);if(!isfinite(f)||(!v->mask[l]&&f!=0))goto bad;
            v->rows[i]=f;norm+=(double)f*f;
        }
        if(v->mask[l]&&(!(norm>0)||!isfinite(norm)||
            (v->mode==XE_STEERING_PROJECTION&&fabs(norm-1)>1e-4)))goto bad;
    }
    format_sha256 h;format_sha256_init(&h);
    static const char domain[]="xenolith-steering-v1";
    format_sha256_update(&h,domain,sizeof domain);
    format_sha256_update(&h,v->manifest,v->manifest_size);
    format_sha256_update(&h,v->payload,v->payload_size);
    format_sha256_final(&h,hash);xe_steering_hex(hash,v->id);
    json_free(j);*out=v;if(error&&cap)error[0]=0;return 1;
nomem: message="out of memory loading steering";
bad: json_free(j);xe_steering_release(v);return fail(error,cap,message);
}
void xe_steering_retain(xe_steering *v){if(v)v->refs++;}
void xe_steering_release(xe_steering *v){if(v&&!--v->refs){free(v->mask);free(v->rows);free(v->manifest);free(v->payload);free(v);}}
const char *xe_steering_id(const xe_steering *v){return v?v->id:NULL;}
xe_steering_mode xe_steering_get_mode(const xe_steering *v){return v?v->mode:0;}
xe_steering_site xe_steering_get_site(const xe_steering *v){return v?v->site:0;}
int xe_steering_layers(const xe_steering *v){return v?v->layers:0;}
int xe_steering_width(const xe_steering *v){return v?v->width:0;}
const uint8_t *xe_steering_model(const xe_steering *v){return v?v->model:NULL;}
int xe_steering_active(const xe_steering *v,int l){return v&&l>=0&&l<v->layers&&v->mask[l];}
const float *xe_steering_row(const xe_steering *v,int l){return xe_steering_active(v,l)?v->rows+(size_t)l*v->width:NULL;}
void xe_steering_apply(float *h,const float *v,int width,xe_steering_mode mode,float strength){
    if(!v||strength==0)return;
    float scale=strength;
    if(mode==XE_STEERING_PROJECTION){float dot=0;for(int i=0;i<width;i++)dot+=h[i]*v[i];scale=-strength*dot;}
    for(int i=0;i<width;i++)h[i]+=scale*v[i];
}
static int write_file(const char *path,const void *data,size_t n){
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);if(fd<0)return 0;
    const uint8_t *p=data;size_t at=0;
    while(at<n){ssize_t r=write(fd,p+at,n-at);if(r<0&&errno==EINTR)continue;if(r<=0)break;at+=(size_t)r;}
    int ok=at==n&&fsync(fd)==0;if(close(fd))ok=0;return ok;
}
int xe_steering_open_id(const char *root,const char *id,int layers,int width,const uint8_t model[32],xe_steering **out,char *error,size_t cap){
    char path[4096];
    if(!root||!xe_steering_valid_id(id)||snprintf(path,sizeof path,"%s/%s/manifest.json",root,id)>=(int)sizeof path)
        return fail(error,cap,"invalid steering artifact identity");
    xe_steering *v=NULL;
    if(!xe_steering_load(path,layers,width,model,&v,error,cap))return 0;
    if(strcmp(id,v->id)){xe_steering_release(v);return fail(error,cap,"steering artifact identity mismatch");}
    *out=v;return 1;
}
int xe_steering_import(const char *root,const char *manifest,int layers,int width,const uint8_t model[32],char id[65],char *error,size_t cap){
    xe_steering *v=NULL;if(!root||!id)return fail(error,cap,"invalid steering import arguments");
    if(!xe_steering_load(manifest,layers,width,model,&v,error,cap))return 0;
    char temp[4096],dest[4096],mp[4096],vp[4096];int ok=0;
    if(snprintf(temp,sizeof temp,"%s/.import-XXXXXX",root)>=(int)sizeof temp||
       snprintf(dest,sizeof dest,"%s/%s",root,v->id)>=(int)sizeof dest||!mkdtemp(temp))goto done;
    if(snprintf(mp,sizeof mp,"%s/manifest.json",temp)>=(int)sizeof mp||
       snprintf(vp,sizeof vp,"%s/vectors.f32",temp)>=(int)sizeof vp)goto cleanup_dir;
    if(write_file(mp,v->manifest,v->manifest_size)&&write_file(vp,v->payload,v->payload_size)){
        int fd=open(temp,O_RDONLY|O_DIRECTORY|O_CLOEXEC);
        int synced=fd>=0&&fsync(fd)==0;if(fd>=0)close(fd);
        if(synced&&rename(temp,dest)==0){
            fd=open(root,O_RDONLY|O_DIRECTORY|O_CLOEXEC);ok=fd>=0&&fsync(fd)==0;if(fd>=0)close(fd);
        }else if(errno==EEXIST||errno==ENOTEMPTY){
            xe_steering *existing=NULL;
            ok=xe_steering_open_id(root,v->id,layers,width,model,&existing,error,cap);
            xe_steering_release(existing);
        }
    }
    unlink(mp);unlink(vp);
cleanup_dir:rmdir(temp);
done:if(ok)memcpy(id,v->id,65);xe_steering_release(v);
    return ok?1:fail(error,cap,"cannot store steering artifact");
}
