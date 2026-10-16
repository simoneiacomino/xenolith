#define _POSIX_C_SOURCE 200809L
#include "vision_internal.h"
#include "format.h"

#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STBI_MAX_DIMENSIONS 16384
#include "vendor/stb/stb_image.h"

static int fail(char *out, size_t cap, const char *message) {
    if (out && cap) snprintf(out, cap, "%s", message);
    return 0;
}

static void hash_u32(format_sha256 *h, uint32_t v) {
    uint8_t b[4]; format_put_u32le(b, v); format_sha256_update(h, b, sizeof b);
}

static void hash_string(format_sha256 *h, const char *s) {
    hash_u32(h, (uint32_t)strlen(s)); format_sha256_update(h, s, strlen(s));
}

int xe_image_size(int w, int h, int budget, int *ow, int *oh) {
    if (w < 1 || h < 1 || w > 16384 || h > 16384 || !ow || !oh ||
        !(budget == 70 || budget == 140 || budget == 280 || budget == 560 || budget == 1120))
        return 0;
    double factor = sqrt((double)budget * 48 * 48 / ((double)w * h));
    int tw = (int)floor(w * factor / 48) * 48;
    int th = (int)floor(h * factor / 48) * 48;
    if (!tw) { tw = 48; th = (h / w < budget ? h / w : budget) * 48; }
    if (!th) { th = 48; tw = (w / h < budget ? w / h : budget) * 48; }
    if (tw < 48 || th < 48 || (uint64_t)tw * th > (uint64_t)budget * 48 * 48)
        return 0;
    *ow = tw; *oh = th;
    return 1;
}

/* Pillow bicubic, a=-0.5, separable antialiasing and 22-bit coefficients. */
static double cubic(double x) {
    x = fabs(x);
    if (x < 1) return (1.5*x - 2.5)*x*x + 1;
    if (x < 2) return (((x-5)*x+8)*x-4)*-0.5;
    return 0;
}

static unsigned char *resize_axis(const unsigned char *src, int w, int h,
                                  int out_size, int vertical) {
    int in_size = vertical ? h : w;
    int ow = vertical ? w : out_size, oh = vertical ? out_size : h;
    unsigned char *out = malloc((size_t)ow * oh * 3);
    double scale = (double)in_size / out_size, fs = scale > 1 ? scale : 1;
    int max_coeff = (int)ceil(2*fs)*2+1;
    int32_t *coeff = malloc((size_t)max_coeff * sizeof(*coeff));
    if (!out || !coeff) { free(out); free(coeff); return NULL; }
    for (int i = 0; i < out_size; i++) {
        double center = (i+0.5)*scale;
        int first = (int)(center-2*fs+0.5), end = (int)(center+2*fs+0.5);
        if (first < 0) first = 0;
        if (end > in_size) end = in_size;
        double total = 0;
        for (int k = first; k < end; k++) total += cubic((k-center+0.5)/fs);
        for (int k = first; k < end; k++) {
            double v = cubic((k-center+0.5)/fs)/total;
            coeff[k-first] = (int32_t)(v*4194304 + (v < 0 ? -0.5 : 0.5));
        }
        for (int j = 0; j < (vertical ? w : h); j++) {
            for (int c = 0; c < 3; c++) {
                int64_t sum = 2097152;
                for (int k = first; k < end; k++) {
                    size_t index = vertical ? ((size_t)k*w+j)*3+c : ((size_t)j*w+k)*3+c;
                    sum += (int64_t)src[index]*coeff[k-first];
                }
                int v = (int)(sum >> 22);
                size_t index = vertical ? ((size_t)i*w+j)*3+c : ((size_t)j*ow+i)*3+c;
                out[index] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
        }
    }
    free(coeff);
    return out;
}

static uint16_t exif_u16(const unsigned char *p,int little) {
    return little?(uint16_t)(p[0]|p[1]<<8):(uint16_t)(p[0]<<8|p[1]);
}
static uint32_t exif_u32(const unsigned char *p,int little) {
    return little?(uint32_t)p[0]|(uint32_t)p[1]<<8|(uint32_t)p[2]<<16|(uint32_t)p[3]<<24:
                  (uint32_t)p[0]<<24|(uint32_t)p[1]<<16|(uint32_t)p[2]<<8|p[3];
}
static int tiff_orientation(const unsigned char *p,size_t n) {
    if(n<8)return 0;
    int little=p[0]=='I'&&p[1]=='I';
    if(!little&&!(p[0]=='M'&&p[1]=='M'))return 0;
    if(exif_u16(p+2,little)!=42)return 0;
    size_t offset=exif_u32(p+4,little);
    if(offset>n-2)return 0;
    unsigned count=exif_u16(p+offset,little);offset+=2;
    if(count>(n-offset)/12)return 0;
    for(unsigned i=0;i<count;i++) {
        const unsigned char *entry=p+offset+i*12;
        if(exif_u16(entry,little)==0x112) {
            if(exif_u16(entry+2,little)!=3||exif_u32(entry+4,little)!=1)return 0;
            int orientation=exif_u16(entry+8,little);return orientation>=1&&orientation<=8?orientation:0;
        }
    }
    return 1;
}
static int image_orientation(const unsigned char *p,size_t n) {
    if(n>=2&&p[0]==255&&p[1]==216) {
        size_t pos=2;
        while(pos+4<=n&&p[pos]==255) {
            while(pos<n&&p[pos]==255)pos++;
            if(pos>=n)return 0;
            int marker=p[pos++];if(marker==218||marker==217)break;
            if(marker==1||(marker>=208&&marker<=215))continue;
            if(pos+2>n)return 0;
            size_t length=(size_t)p[pos]<<8|p[pos+1];
            if(length<2||length>n-pos)return 0;
            if(marker==225&&length>=8&&!memcmp(p+pos+2,"Exif\0\0",6))
                return tiff_orientation(p+pos+8,length-8);
            pos+=length;
        }
    } else if(n>=8&&!memcmp(p,"\x89PNG\r\n\x1a\n",8)) {
        size_t pos=8;
        while(n-pos>=12) {
            size_t length=exif_u32(p+pos,0);if(length>n-pos-12)return 0;
            if(!memcmp(p+pos+4,"eXIf",4))return tiff_orientation(p+pos+8,length);
            pos+=length+12;
        }
    }
    return 1;
}
static unsigned char *orient_rgb(unsigned char *rgb,int *width,int *height,int orientation) {
    if(orientation==1)return rgb;
    int w=*width,h=*height,ow=orientation>=5?h:w,oh=orientation>=5?w:h;
    unsigned char *out=malloc((size_t)ow*oh*3);if(!out)return NULL;
    for(int y=0;y<h;y++)for(int x=0;x<w;x++) {
        int dx=x,dy=y;
        switch(orientation) {
        case 2:dx=w-1-x;break;
        case 3:dx=w-1-x;dy=h-1-y;break;
        case 4:dy=h-1-y;break;
        case 5:dx=y;dy=x;break;
        case 6:dx=h-1-y;dy=x;break;
        case 7:dx=h-1-y;dy=w-1-x;break;
        case 8:dx=y;dy=w-1-x;break;
        }
        memcpy(out+((size_t)dy*ow+dx)*3,rgb+((size_t)y*w+x)*3,3);
    }
    free(rgb);*width=ow;*height=oh;return out;
}

int xe_image_load(xe_image *out, const char *path, int budget, char *error, size_t cap) {
    if (!out || !path) return fail(error, cap, "invalid image argument");
    memset(out, 0, sizeof *out);
    FILE *f = fopen(path, "rb");
    if (!f) return fail(error, cap, "cannot open image");
    uint64_t bytes;
    if (!format_file_size(f, &bytes) || bytes < 8 || bytes > 64*UINT64_C(1024)*1024) {
        fclose(f); return fail(error, cap, "image byte limit exceeded");
    }
    unsigned char *encoded = malloc((size_t)bytes);
    if (!encoded || !format_read(f, encoded, (size_t)bytes)) {
        free(encoded); fclose(f); return fail(error, cap, "cannot read image");
    }
    fclose(f);
    int ok = xe_image_load_memory(out, encoded, (size_t)bytes, budget, error, cap);
    free(encoded);
    return ok;
}

int xe_image_load_memory(xe_image *out, const unsigned char *encoded, size_t bytes,
                         int budget, char *error, size_t cap) {
    if (!out || !encoded || bytes < 8 || bytes > 64*UINT64_C(1024)*1024)
        return fail(error, cap, "invalid image bytes");
    memset(out, 0, sizeof *out);
    int w, h, channels, tw, th;
    int orientation=image_orientation(encoded,bytes);
    if(!orientation)return fail(error,cap,"invalid EXIF orientation metadata");
    if (!stbi_info_from_memory(encoded, (int)bytes, &w, &h, &channels) ||
        (uint64_t)w*h > 32*UINT64_C(1024)*1024 ||
        !xe_image_size(w, h, budget, &tw, &th)) {
        return fail(error, cap, "invalid image dimensions or budget");
    }
    format_sha256_bytes(encoded, (size_t)bytes, out->encoded_hash);
    unsigned char *rgba = stbi_load_from_memory(encoded, (int)bytes, &w, &h, &channels, 4);
    if (!rgba) return fail(error, cap, "cannot decode PNG/JPEG");
    unsigned char *rgb = malloc((size_t)w*h*3);
    if (!rgb) { stbi_image_free(rgba); return fail(error, cap, "image allocation failed"); }
    for (size_t i = 0; i < (size_t)w*h; i++)
        for (int c = 0; c < 3; c++)
            rgb[i*3+c] = (unsigned char)((rgba[i*4+c]*rgba[i*4+3]+255*(255-rgba[i*4+3])+127)/255);
    stbi_image_free(rgba);
    unsigned char *oriented=orient_rgb(rgb,&w,&h,orientation);
    if(!oriented){free(rgb);return fail(error,cap,"orientation allocation failed");}
    rgb=oriented;
    if(!xe_image_size(w,h,budget,&tw,&th)){free(rgb);return fail(error,cap,"invalid oriented image dimensions");}
    unsigned char *horizontal = resize_axis(rgb, w, h, tw, 0);
    unsigned char *resized = horizontal ? resize_axis(horizontal, tw, h, th, 1) : NULL;
    free(horizontal);
    if (!resized) { free(rgb); return fail(error, cap, "resize allocation failed"); }
    out->encoded = malloc(bytes);
    if (!out->encoded) { free(rgb); free(resized); return fail(error, cap, "image allocation failed"); }
    memcpy(out->encoded, encoded, bytes); out->encoded_size = bytes;
    out->rgb = rgb; out->resized = resized; out->width = w; out->height = h;
    out->resized_width = tw; out->resized_height = th;
    out->budget = budget; out->rows = tw/48*(th/48);
    format_sha256 hash; format_sha256_init(&hash);
    hash_u32(&hash, w); hash_u32(&hash, h);
    format_sha256_update(&hash, rgb, (size_t)w*h*3);
    format_sha256_final(&hash, out->rgb_hash);
    format_sha256_bytes(resized,(size_t)tw*th*3,out->resized_hash);
    xe_image_refresh_identity(out);
    return 1;
}

void xe_image_refresh_identity(xe_image *im) {
    format_sha256 hash;
    format_sha256_init(&hash); hash_string(&hash, "xe-image/v1-exif-rgb-white-hf-budget-pillow-bicubic");
    format_sha256_update(&hash, im->encoded_hash, 32);
    format_sha256_update(&hash, im->rgb_hash, 32);
    format_sha256_update(&hash, im->resized_hash, 32);
    hash_u32(&hash, im->budget); hash_u32(&hash, im->resized_width); hash_u32(&hash, im->resized_height);
    hash_u32(&hash, im->rows); format_sha256_final(&hash, im->identity);
}

void xe_image_free(xe_image *image) {
    if (image) { free(image->encoded); free(image->rgb); free(image->resized); memset(image, 0, sizeof *image); }
}

int xe_input_layout_valid(const xe_input *in, int context) {
    if (!in || !in->tokens || in->count < 1 || in->count > context ||
        (in->visual_count && !in->visual) || in->visual_count > (size_t)in->count) return 0;
    int end = 0;size_t media_total=0;
    for (size_t i = 0; i < in->visual_count; i++) {
        const xe_visual_span *v = &in->visual[i];
        if (v->start < end || v->rows < 1 || v->rows > XE_VISUAL_MAX ||
            v->start > in->count-v->rows || !v->embeddings ||
            v->media_size > 64*UINT64_C(1024)*1024 || (v->media_size && !v->media)) return 0;
        media_total+=v->media_size;
        if(media_total>256*UINT64_C(1024)*1024)return 0;
        end = v->start+v->rows;
    }
    return 1;
}

int xe_input_valid(const xe_input *in,int context) {
    if(!xe_input_layout_valid(in,context))return 0;
    for(size_t i=0;i<in->visual_count;i++)
        for(size_t j=0;j<(size_t)in->visual[i].rows*XE_VISUAL_WIDTH;j++)
            if(!isfinite(in->visual[i].embeddings[j]))return 0;
    return 1;
}

int xe_input_boundary(const xe_input *in, int p) {
    if (!in || p < 0 || p > in->count) return 0;
    for (size_t i = 0; i < in->visual_count; i++)
        if (p > in->visual[i].start && p < in->visual[i].start+in->visual[i].rows) return 0;
    return 1;
}

void xe_input_hash(const xe_input *in, uint8_t out[32]) {
    format_sha256 h; format_sha256_init(&h); hash_string(&h, "xe-input/v1");
    hash_u32(&h, in->count);
    for (int i = 0; i < in->count; i++) hash_u32(&h, (uint32_t)in->tokens[i]);
    hash_u32(&h, (uint32_t)in->visual_count);
    for (size_t i = 0; i < in->visual_count; i++) {
        hash_u32(&h, in->visual[i].start); hash_u32(&h, in->visual[i].rows);
        const xe_visual_span *v = in->visual+i;
        format_sha256_update(&h, v->identity, 32);
        hash_u32(&h, v->budget); hash_u32(&h, (uint32_t)v->media_size);
        if (v->media_size) format_sha256_update(&h, v->media, v->media_size);
        for (size_t j = 0; j < (size_t)v->rows*XE_VISUAL_WIDTH; j++) {
            uint32_t b; memcpy(&b, v->embeddings+j, 4); hash_u32(&h, b);
        }
    }
    format_sha256_final(&h, out);
}

int xe_input_copy(xe_owned_input *out, const xe_input *in) {
    if (!out || !xe_input_valid(in, 131072)) return 0;
    size_t size = (size_t)in->count*4;
    size = (size+7)&~(size_t)7;
    size += in->visual_count*sizeof(xe_visual_span);
    for (size_t i=0; i<in->visual_count; i++) {
        size_t extra=(size_t)in->visual[i].rows*XE_VISUAL_WIDTH*4+in->visual[i].media_size;
        extra=(extra+7)&~(size_t)7;
        if (extra > SIZE_MAX-size) return 0;
        size += extra;
    }
    unsigned char *storage=malloc(size);
    if (!storage) return 0;
    xe_owned_input next={.input=*in,.storage=storage};
    memcpy(storage,in->tokens,(size_t)in->count*4);next.input.tokens=(int32_t *)storage;
    size_t offset=((size_t)in->count*4+7)&~(size_t)7;
    xe_visual_span *spans=(xe_visual_span *)(storage+offset);
    next.input.visual=spans;offset+=in->visual_count*sizeof(*spans);
    for (size_t i=0; i<in->visual_count; i++) {
        spans[i]=in->visual[i];size_t bytes=(size_t)spans[i].rows*XE_VISUAL_WIDTH*4;
        spans[i].embeddings=(float *)(storage+offset);
        memcpy(storage+offset,in->visual[i].embeddings,bytes);offset+=bytes;
        spans[i].media=spans[i].media_size ? storage+offset : NULL;
        if (spans[i].media_size) memcpy(storage+offset,in->visual[i].media,spans[i].media_size);
        offset=(offset+spans[i].media_size+7)&~(size_t)7;
    }
    *out=next;return 1;
}

void xe_input_free(xe_owned_input *in) {
    if (in) { free(in->storage);memset(in,0,sizeof *in); }
}

/* Input archive v1: 64-byte header, LE tokens, then 64-byte span headers,
 * original encoded media and LE F32 embeddings. SHA-256 covers all content. */
int xe_input_save(FILE *f,const xe_input *in) {
    if(!f||!xe_input_valid(in,131072))return 0;
    for(size_t i=0;i<in->visual_count;i++)if(!in->visual[i].media_size)return 0;
    uint8_t header[64]={0};memcpy(header,"XEVINPUT",8);
    format_put_u32le(header+8,1);format_put_u32le(header+12,in->count);
    format_put_u32le(header+16,(uint32_t)in->visual_count);xe_input_hash(in,header+32);
    if(!format_seek(f,0)||!format_write(f,header,64))return 0;
    for(int i=0;i<in->count;i++){uint8_t b[4];format_put_u32le(b,in->tokens[i]);if(!format_write(f,b,4))return 0;}
    for(size_t i=0;i<in->visual_count;i++) {
        const xe_visual_span *v=in->visual+i;uint8_t b[64]={0};
        format_put_u32le(b,v->start);format_put_u32le(b+4,v->rows);
        format_put_u32le(b+8,v->budget);format_put_u64le(b+16,v->media_size);
        memcpy(b+32,v->identity,32);
        if(!format_write(f,b,64)||!format_write(f,v->media,v->media_size))return 0;
        for(size_t j=0;j<(size_t)v->rows*XE_VISUAL_WIDTH;j++) {
            uint32_t bits;memcpy(&bits,v->embeddings+j,4);format_put_u32le(b,bits);
            if(!format_write(f,b,4))return 0;
        }
    }
    off_t end=ftello(f);
    return end>=0 && fflush(f)==0 && ftruncate(fileno(f),end)==0;
}

int xe_input_load(FILE *f,int context,xe_owned_input *out) {
    uint8_t header[64];uint64_t size;
    if(!f||!out||!format_file_size(f,&size)||size>UINT64_C(2)*1024*1024*1024||
       !format_seek(f,0)||!format_read(f,header,64)||memcmp(header,"XEVINPUT",8)||
       format_get_u32le(header+8)!=1)return 0;
    for(int i=20;i<32;i++)if(header[i])return 0;
    uint32_t count=format_get_u32le(header+12),n=format_get_u32le(header+16);
    if(context<1||context>131072||!count||count>(uint32_t)context||n>count)return 0;
    int32_t *tokens=malloc((size_t)count*4);xe_visual_span *spans=calloc(n?n:1,sizeof *spans);
    int ok=tokens&&spans;
    for(uint32_t i=0;ok&&i<count;i++) {uint8_t b[4];ok=format_read(f,b,4);if(ok)tokens[i]=(int32_t)format_get_u32le(b);}
    size_t media_total=0;
    for(uint32_t i=0;ok&&i<n;i++) {
        uint8_t b[64];ok=format_read(f,b,64);if(!ok)break;
        xe_visual_span *v=spans+i;uint64_t bytes=format_get_u64le(b+16);
        v->start=(int)format_get_u32le(b);v->rows=(int)format_get_u32le(b+4);
        v->budget=(int)format_get_u32le(b+8);memcpy(v->identity,b+32,32);
        if(v->rows<1||v->rows>1120||!bytes||bytes>64*UINT64_C(1024)*1024||
           media_total+bytes>256*UINT64_C(1024)*1024||format_get_u32le(b+12)||format_get_u64le(b+24)) {ok=0;break;}
        media_total+=(size_t)bytes;v->media_size=(size_t)bytes;
        unsigned char *media=malloc(v->media_size);float *embedding=malloc((size_t)v->rows*XE_VISUAL_WIDTH*4);
        v->media=media;v->embeddings=embedding;ok=media&&embedding&&format_read(f,media,v->media_size);
        for(size_t j=0;ok&&j<(size_t)v->rows*XE_VISUAL_WIDTH;j++) {
            ok=format_read(f,b,4);if(ok){uint32_t bits=format_get_u32le(b);memcpy(embedding+j,&bits,4);}
        }
    }
    xe_input in={tokens,(int)count,spans,n};uint8_t hash[32];
    if(ok)ok=ftello(f)>=0 && (uint64_t)ftello(f)==size && xe_input_valid(&in,context);
    if(ok){xe_input_hash(&in,hash);ok=!memcmp(hash,header+32,32);}
    xe_owned_input next={0};if(ok)ok=xe_input_copy(&next,&in);
    if(spans)for(uint32_t i=0;i<n;i++){free((void *)spans[i].media);free((void *)spans[i].embeddings);}
    free(spans);free(tokens);if(ok)*out=next;return ok;
}

void xe_visual_identity(const xe_image *im, const uint8_t model[32],
                         const uint8_t projector[32], const float *embedding,
                         const char *revision, uint8_t out[32]) {
    format_sha256 h; format_sha256_init(&h); hash_string(&h, "xe-visual/v1");
    format_sha256_update(&h, im->identity, 32);
    format_sha256_update(&h, model, 32); format_sha256_update(&h, projector, 32);
    hash_string(&h, revision);
    for (size_t i = 0; i < (size_t)im->rows*XE_VISUAL_WIDTH; i++) {
        uint32_t b; memcpy(&b, embedding+i, 4); hash_u32(&h, b);
    }
    format_sha256_final(&h, out);
}

typedef struct { const uint8_t *p, *end; int bad; } cursor;
static const uint8_t *take(cursor *c, uint64_t n) {
    if (c->bad || n > (uint64_t)(c->end-c->p)) { c->bad=1; return NULL; }
    const uint8_t *p=c->p; c->p+=(size_t)n; return p;
}
static uint32_t u32(cursor *c) { const uint8_t *p=take(c,4);return p?format_get_u32le(p):0; }
static uint64_t u64(cursor *c) { const uint8_t *p=take(c,8);return p?format_get_u64le(p):0; }
static int str(cursor *c, char *out, size_t cap) {
    uint64_t n=u64(c);const uint8_t *p=take(c,n);
    if (!p || n >= cap || memchr(p,0,(size_t)n)) { c->bad=1;return 0; }
    memcpy(out,p,(size_t)n);out[n]=0;return 1;
}
static void skip(cursor *c,uint32_t type,int depth) {
    if (depth>1) { c->bad=1;return; }
    static const int size[]={1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (type==8) { uint64_t n=u64(c);take(c,n); }
    else if(type==9) { uint32_t t=u32(c);uint64_t n=u64(c);if(n>1048576){c->bad=1;return;}for(uint64_t i=0;i<n&&!c->bad;i++)skip(c,t,depth+1); }
    else if(type<13) take(c,size[type]);
    else c->bad=1;
}

const xe_vision_tensor *xe_vision_tensor_get(const xe_vision_model *m,const char *name) {
    for(size_t i=0;i<m->tensor_count;i++)if(!strcmp(m->tensors[i].name,name))return m->tensors+i;
    return NULL;
}

void xe_vision_model_close(xe_vision_model *m) {
    if(!m)return;
    if(m->mapping)munmap(m->mapping,m->mapping_size);
    free(m->tensors);memset(m,0,sizeof *m);
}

/* The manifest identifies the tested pair. Compatible variants get different
 * cache identities; matching these hashes is not a loading requirement. */
int xe_vision_weights_reference(const uint8_t id[32], int projector) {
    static const uint8_t pins[2][32] = {
        {0xa7,0xc5,0xbc,0x71,0x5f,0x5f,0xf8,0xe9,0x9a,0x3e,0x89,0x01,0xce,0x7d,0x2b,0x42,0xb4,0x02,0xc6,0x69,0xbf,0x24,0xf7,0xc5,0x25,0x07,0x47,0x63,0x3d,0x0f,0x58,0x91},
        {0xd0,0x0f,0x21,0x1a,0x7d,0x4f,0x7f,0xb1,0x9b,0xd9,0xb7,0x5d,0x8e,0x93,0x42,0xec,0xcf,0xfb,0x59,0x20,0xb0,0x8f,0xd9,0x56,0x21,0x67,0x56,0x0f,0xdf,0xcc,0x5d,0xd1}
    };
    return id && !memcmp(id, pins[!!projector], 32);
}

static int vision_tensor_shape(const xe_vision_model *m, const char *name,
                                uint32_t type, uint32_t dims,
                                uint64_t a, uint64_t b, uint64_t c, uint64_t d) {
    const xe_vision_tensor *t=xe_vision_tensor_get(m,name);
    const uint64_t ne[4]={a,b,c,d};
    if(!t || t->type!=type || t->dimensions!=dims)return 0;
    for(uint32_t i=0;i<dims;i++)if(t->ne[i]!=ne[i])return 0;
    return 1;
}

/* Every tensor read by the fixed encoder must have its exact type and shape.
 * A content checksum cannot replace these checks for compatible fine-tunes. */
static int vision_tensor_schema(const xe_vision_model *m) {
    if(!vision_tensor_shape(m,"mm.input_projection.weight",1,2,1152,2816,0,0) ||
       !vision_tensor_shape(m,"v.patch_embd.weight",1,4,16,16,3,1152) ||
       !vision_tensor_shape(m,"v.position_embd.weight",0,3,1152,10240,2,0) ||
       !vision_tensor_shape(m,"v.std_bias",0,1,1152,0,0,0) ||
       !vision_tensor_shape(m,"v.std_scale",0,1,1152,0,0,0))return 0;
    static const struct { const char *name; uint32_t type; uint64_t a,b; } layer[] = {
        {"ln1.weight",0,1152,0}, {"ln2.weight",0,1152,0},
        {"attn_post_norm.weight",0,1152,0}, {"ffn_post_norm.weight",0,1152,0},
        {"attn_q_norm.weight",0,72,0}, {"attn_k_norm.weight",0,72,0},
        {"attn_q.weight",1,1152,1152}, {"attn_k.weight",1,1152,1152},
        {"attn_v.weight",1,1152,1152}, {"attn_out.weight",1,1152,1152},
        {"ffn_gate.weight",1,1152,4304}, {"ffn_up.weight",1,1152,4304},
        {"ffn_down.weight",1,4304,1152}
    };
    for(int i=0;i<27;i++)for(size_t j=0;j<sizeof layer/sizeof *layer;j++) {
        char name[128];snprintf(name,sizeof name,"v.blk.%d.%s",i,layer[j].name);
        if(!vision_tensor_shape(m,name,layer[j].type,layer[j].b?2:1,
                                layer[j].a,layer[j].b,0,0))return 0;
    }
    return 1;
}

static int vision_metadata(cursor *c,uint64_t count,uint32_t *alignment) {
    static const struct { const char *name; uint32_t value; } integers[] = {
        {"clip.vision.projection_dim",2816}, {"clip.vision.patch_size",16},
        {"clip.vision.embedding_length",1152}, {"clip.vision.feed_forward_length",4304},
        {"clip.vision.block_count",27}, {"clip.vision.attention.head_count",16},
        {"clip.vision.image_size",224}
    };
    uint32_t seen=0;
    for(uint64_t i=0;i<count&&!c->bad;i++) {
        char key[256],value[256];if(!str(c,key,sizeof key))return 0;
        uint32_t type=u32(c),bit=0;
        size_t j;
        for(j=0;j<sizeof integers/sizeof *integers;j++)if(!strcmp(key,integers[j].name))break;
        if(j<sizeof integers/sizeof *integers) {
            bit=1u<<j;if(type!=4||u32(c)!=integers[j].value)return 0;
        } else if(!strcmp(key,"general.architecture") || !strcmp(key,"clip.vision.projector_type")) {
            int arch=!strcmp(key,"general.architecture");bit=1u<<(arch?7:8);
            if(type!=8||!str(c,value,sizeof value)||strcmp(value,arch?"clip":"gemma4v"))return 0;
        } else if(!strcmp(key,"clip.vision.image_mean")||!strcmp(key,"clip.vision.image_std")) {
            int std=!strcmp(key,"clip.vision.image_std");bit=1u<<(std?10:9);
            if(type!=9||u32(c)!=6||u64(c)!=3)return 0;
            for(int k=0;k<3;k++){uint32_t bits=u32(c);float f;memcpy(&f,&bits,4);if(f!=(std?1.0f:0.0f))return 0;}
        } else if(!strcmp(key,"clip.vision.attention.layer_norm_epsilon")) {
            bit=1u<<11;if(type!=6)return 0;
            uint32_t bits=u32(c);float f;memcpy(&f,&bits,4);if(f!=1e-6f)return 0;
        } else if(!strcmp(key,"clip.has_vision_encoder")||!strcmp(key,"clip.use_gelu")) {
            bit=1u<<(!strcmp(key,"clip.has_vision_encoder")?12:13);
            if(type!=7)return 0;
            const uint8_t *v=take(c,1);if(!v||*v!=1)return 0;
        } else if(!strcmp(key,"general.alignment")) {
            bit=1u<<14;if(type!=4)return 0;*alignment=u32(c);
            if(*alignment<4||(*alignment&(*alignment-1)))return 0;
        } else skip(c,type,0);
        if(bit&&(seen&bit))return 0;
        seen|=bit;
    }
    /* clip.use_gelu is absent in the reference GGUF; Gemma4v uses GELU. */
    return !c->bad&&(seen&0x1fff)==0x1fff;
}

int xe_vision_model_open(xe_vision_model *m,const char *path,char *error,size_t cap) {
    if(!m||!path)return fail(error,cap,"invalid mmproj argument");
    memset(m,0,sizeof *m);int fd=open(path,O_RDONLY);struct stat st;
    if(fd<0)return fail(error,cap,"cannot open mmproj");
    if(fstat(fd,&st)||st.st_size<24||(uint64_t)st.st_size>SIZE_MAX){close(fd);return fail(error,cap,"invalid mmproj size");}
    m->mapping_size=(size_t)st.st_size;m->mapping=mmap(NULL,m->mapping_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(m->mapping==MAP_FAILED){m->mapping=NULL;return fail(error,cap,"cannot map mmproj");}
    cursor c={m->mapping,(const uint8_t *)m->mapping+m->mapping_size,0};
    if(u32(&c)!=UINT32_C(0x46554747)||u32(&c)!=3)goto bad;
    uint64_t nt=u64(&c),nk=u64(&c);if(nt!=356||nk>4096)goto bad;
    uint32_t alignment=32;
    if(!vision_metadata(&c,nk,&alignment))goto bad;
    m->tensors=calloc((size_t)nt,sizeof(*m->tensors));if(!m->tensors)goto bad;
    m->tensor_count=(size_t)nt;
    for(size_t i=0;i<m->tensor_count&&!c.bad;i++) {
        xe_vision_tensor *t=m->tensors+i;if(!str(&c,t->name,sizeof t->name))goto bad;
        t->dimensions=u32(&c);if(!t->dimensions||t->dimensions>4)goto bad;
        uint64_t elements=1;
        for(uint32_t j=0;j<t->dimensions;j++){t->ne[j]=u64(&c);if(!t->ne[j]||!format_mul_u64(elements,t->ne[j],&elements))goto bad;}
        t->type=u32(&c);t->offset=u64(&c);if(t->type>1||!format_mul_u64(elements,t->type?2:4,&t->bytes))goto bad;
        for(size_t j=0;j<i;j++)if(!strcmp(t->name,m->tensors[j].name))goto bad;
    }
    uint64_t offset;if(c.bad||!format_align_u64((uint64_t)(c.p-(const uint8_t *)m->mapping),alignment,&offset)||offset>m->mapping_size)goto bad;
    m->data_offset=(size_t)offset;
    for(size_t i=0;i<m->tensor_count;i++) {
        xe_vision_tensor *t=m->tensors+i;
        if(t->offset%alignment||t->offset>m->mapping_size-offset||t->bytes>m->mapping_size-offset-t->offset)goto bad;
        for(size_t j=0;j<i;j++) {
            const xe_vision_tensor *other=m->tensors+j;
            if(t->offset<other->offset+other->bytes && other->offset<t->offset+t->bytes)goto bad;
        }
        t->data=(const uint8_t *)m->mapping+offset+t->offset;
    }
    if(!vision_tensor_schema(m))goto bad;
    format_sha256_bytes(m->mapping,m->mapping_size,m->sha256);
    if(!xe_vision_weights_reference(m->sha256,1))
        fprintf(stderr,"xenolith: vision projector differs from the reference weights; "
                       "using its full content identity for cache isolation.\n");
    return 1;
bad:
    xe_vision_model_close(m);return fail(error,cap,"mmproj schema mismatch");
}

int xe_image_record_parse(const void *data,size_t bytes,xe_image *meta,
                           int *offset,const uint8_t **media,size_t *size) {
    if(!data||!meta||!offset||!media||!size||bytes<XE_IMAGE_RECORD_HEADER)return 0;
    const uint8_t *p=data;
    if(memcmp(p,"XEVIMG1\n",8)||format_get_u32le(p+36)!=XE_IMAGE_RECORD_HEADER)return 0;
    uint64_t length=format_get_u64le(p+40);
    if(!length||length>XE_IMAGE_RECORD_LIMIT||length!=bytes-XE_IMAGE_RECORD_HEADER)return 0;
    for(int i=272;i<XE_IMAGE_RECORD_HEADER;i++)if(p[i])return 0;
    memset(meta,0,sizeof *meta);
    *offset=(int)format_get_u32le(p+8);meta->rows=(int)format_get_u32le(p+12);
    meta->budget=(int)format_get_u32le(p+16);meta->width=(int)format_get_u32le(p+20);
    meta->height=(int)format_get_u32le(p+24);meta->resized_width=(int)format_get_u32le(p+28);
    meta->resized_height=(int)format_get_u32le(p+32);
    if(*offset<0||meta->rows<1||meta->rows>1120||!(meta->budget==70||meta->budget==140||meta->budget==280||meta->budget==560||meta->budget==1120)||meta->rows>meta->budget||
       meta->width<1||meta->height<1||meta->width>16384||meta->height>16384||
       (uint64_t)meta->width*meta->height>32*UINT64_C(1024)*1024||
       meta->resized_width<48||meta->resized_height<48||meta->resized_width%48||meta->resized_height%48||
       (uint64_t)(meta->resized_width/48)*(meta->resized_height/48)!=(uint64_t)meta->rows)return 0;
    memcpy(meta->encoded_hash,p+48,32);memcpy(meta->rgb_hash,p+80,32);
    memcpy(meta->resized_hash,p+240,32);xe_image_refresh_identity(meta);
    *media=p+XE_IMAGE_RECORD_HEADER;*size=(size_t)length;return 1;
}
