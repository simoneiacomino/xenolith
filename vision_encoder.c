/* Vision encoder: borrows the engine's GPU execution resources. */
#include "vision_encoder.h"
#include "format.h"

#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    xe_engine *owner;
    xe_vision_gpu *gpu;
    const xe_vision_model *model;
} vision_encoder;

static _Noreturn void xe_fatal(const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "xenolith: ");
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fprintf(stderr, "\n");
    exit(1);
}

static void xe_ze_check(const char *operation, ze_result_t result) {
    if (result != ZE_RESULT_SUCCESS) xe_fatal("%s failed: 0x%x", operation, result);
}

static void xe_gpu_pointer_arg(ze_kernel_handle_t kernel, uint32_t index,
                               const void *pointer) {
    xe_ze_check("zeKernelSetArgumentValue pointer",
                zeKernelSetArgumentValue(kernel, index, sizeof(pointer), &pointer));
}

static void xe_gpu_int_arg(ze_kernel_handle_t kernel, uint32_t index, int value) {
    xe_ze_check("zeKernelSetArgumentValue int",
                zeKernelSetArgumentValue(kernel, index, sizeof(value), &value));
}

static ze_kernel_handle_t xe_vision_kernel(vision_encoder *e, int index) {
    static const char *names[7] = {
        "xe_vision_dense", "xe_vision_norm", "xe_vision_head", "xe_vision_attention",
        "xe_vision_add", "xe_vision_geglu", "xe_vision_pool"
    };
    if (!e->gpu->kernels[index]) {
        ze_kernel_desc_t desc = { .stype=ZE_STRUCTURE_TYPE_KERNEL_DESC, .pKernelName=names[index] };
        xe_ze_check("zeKernelCreate vision", zeKernelCreate(e->gpu->module,&desc,&e->gpu->kernels[index]));
        xe_ze_check("zeKernelSetGroupSize vision", zeKernelSetGroupSize(e->gpu->kernels[index],128,1,1));
    }
    return e->gpu->kernels[index];
}

static void xe_vision_launch(vision_encoder *e, ze_kernel_handle_t kernel, int gx, int gy) {
    ze_group_count_t groups={(uint32_t)gx,(uint32_t)gy,1};
    xe_ze_check("zeCommandListAppendLaunchKernel vision",
                zeCommandListAppendLaunchKernel(e->gpu->commands,kernel,&groups,NULL,0,NULL));
    /* Production must bound queued GPU work without relying on dump I/O. */
    xe_ze_check("zeCommandListHostSynchronize vision dispatch",
                zeCommandListHostSynchronize(e->gpu->commands,UINT64_MAX));
}

static const void *xe_vision_weight(vision_encoder *e,const char *name,int type) {
    const xe_vision_tensor *t=xe_vision_tensor_get(e->model,name);
    if(!t||t->type!=(uint32_t)type)xe_fatal("vision: missing or invalid tensor %s",name);
    return t->data;
}

static void xe_vision_dense(vision_encoder *e,const float *x,const char *weight,
                            float *out,int rows,int width,int columns) {
    ze_kernel_handle_t k=xe_vision_kernel(e,0);
    xe_gpu_pointer_arg(k,0,x);xe_gpu_pointer_arg(k,1,xe_vision_weight(e,weight,1));xe_gpu_pointer_arg(k,2,out);
    xe_gpu_int_arg(k,3,rows);xe_gpu_int_arg(k,4,width);xe_gpu_int_arg(k,5,columns);
    xe_vision_launch(e,k,(rows+15)/16,(columns+15)/16);
}

static void xe_vision_norm(vision_encoder *e,const float *x,const char *weight,
                           float *out,int rows,int width) {
    ze_kernel_handle_t k=xe_vision_kernel(e,1);
    xe_gpu_pointer_arg(k,0,x);xe_gpu_pointer_arg(k,1,weight?xe_vision_weight(e,weight,0):NULL);xe_gpu_pointer_arg(k,2,out);
    xe_gpu_int_arg(k,3,rows);xe_gpu_int_arg(k,4,width);xe_vision_launch(e,k,rows,1);
}

static void xe_vision_head(vision_encoder *e,float *x,const char *weight,int rows,int columns,int rotary) {
    ze_kernel_handle_t k=xe_vision_kernel(e,2);
    xe_gpu_pointer_arg(k,0,x);xe_gpu_pointer_arg(k,1,weight?xe_vision_weight(e,weight,0):NULL);
    xe_gpu_int_arg(k,2,rows);xe_gpu_int_arg(k,3,columns);xe_gpu_int_arg(k,4,rotary);
    xe_vision_launch(e,k,rows*16,1);
}

static void xe_vision_binary(vision_encoder *e,int index,float *x,const float *y,int elements) {
    ze_kernel_handle_t k=xe_vision_kernel(e,index);
    xe_gpu_pointer_arg(k,0,x);xe_gpu_pointer_arg(k,1,y);xe_gpu_int_arg(k,2,elements);
    xe_vision_launch(e,k,(elements+127)/128,1);
}

static void xe_vision_wait(vision_encoder *e) {
    xe_ze_check("zeCommandListHostSynchronize vision",zeCommandListHostSynchronize(e->gpu->commands,UINT64_MAX));
}

static void xe_vision_dump(vision_encoder *e,const char *directory,const char *name,
                           const float *x,size_t elements) {
    if(!directory)return;
    xe_vision_wait(e);
    char path[4096];int n=snprintf(path,sizeof path,"%s/%s.f32",directory,name);
    if(n<0||(size_t)n>=sizeof path)xe_fatal("vision dump path too long");
    FILE *f=fopen(path,"wb");if(!f)xe_fatal("cannot open vision dump %s",path);
    int ok=format_write(f,x,elements*sizeof(float));if(fclose(f))ok=0;
    if(!ok)xe_fatal("cannot write vision dump %s",path);
}

int xe_vision_encode(xe_engine *owner, xe_vision_gpu *gpu,
                     const xe_vision_model *model, const xe_image *im,
                     float *output, const char *dump) {
    vision_encoder encoder = { .owner=owner, .gpu=gpu, .model=model };
    vision_encoder *e = &encoder;
    int cols=im->resized_width/16,rows=im->resized_height/16,n=rows*cols;
    if(cols%3||rows%3||n!=im->rows*9)return 0;
    size_t wide=(size_t)n*1152,ff=(size_t)n*4304;
    float *patch=xe_vision_alloc_shared(e->owner,(size_t)n*768*sizeof(float));
    for(int py=0;py<rows;py++)for(int px=0;px<cols;px++)
        for(int c=0;c<3;c++)for(int y=0;y<16;y++)for(int x=0;x<16;x++) {
            unsigned char v=im->resized[((size_t)(py*16+y)*im->resized_width+px*16+x)*3+c];
            patch[((size_t)py*cols+px)*768+c*256+y*16+x]=((float)v/255)*2-1;
        }
    float *hidden=xe_vision_alloc_shared(e->owner,wide*4),*norm=xe_vision_alloc_shared(e->owner,wide*4);
    float *q=xe_vision_alloc_shared(e->owner,wide*4),*k=xe_vision_alloc_shared(e->owner,wide*4),*v=xe_vision_alloc_shared(e->owner,wide*4);
    float *attention=xe_vision_alloc_shared(e->owner,wide*4),*projected=xe_vision_alloc_shared(e->owner,wide*4);
    float *gate=xe_vision_alloc_shared(e->owner,ff*4),*up=xe_vision_alloc_shared(e->owner,ff*4);
    xe_vision_dense(e,patch,"v.patch_embd.weight",hidden,n,768,1152);xe_vision_wait(e);
    xe_vision_free_shared(e->owner,patch);
    const float *pos=xe_vision_weight(e,"v.position_embd.weight",0);
    for(int row=0;row<n;row++)for(int d=0;d<1152;d++)
        hidden[(size_t)row*1152+d]+=pos[(size_t)(row%cols)*1152+d]+pos[(size_t)(10240+row/cols)*1152+d];
    xe_vision_dump(e,dump,"pos_embd",hidden,wide);
    for(int layer=0;layer<27;layer++) {
        char name[128];
#define VW(suffix) (snprintf(name,sizeof name,"v.blk.%d.%s",layer,suffix),name)
        xe_vision_norm(e,hidden,VW("ln1.weight"),norm,n,1152);
        xe_vision_dense(e,norm,VW("attn_q.weight"),q,n,1152,1152);
        xe_vision_dense(e,norm,VW("attn_k.weight"),k,n,1152,1152);
        xe_vision_dense(e,norm,VW("attn_v.weight"),v,n,1152,1152);
        xe_vision_head(e,q,VW("attn_q_norm.weight"),n,cols,1);
        xe_vision_head(e,k,VW("attn_k_norm.weight"),n,cols,1);
        xe_vision_head(e,v,NULL,n,cols,0);
        ze_kernel_handle_t att=xe_vision_kernel(e,3);
        xe_gpu_pointer_arg(att,0,q);xe_gpu_pointer_arg(att,1,k);xe_gpu_pointer_arg(att,2,v);xe_gpu_pointer_arg(att,3,attention);
        xe_gpu_int_arg(att,4,n);
        /* Tile queries only: every query still reads every image key/value.
         * A monolithic 10080-patch dispatch can trigger WSL device loss. */
        for(int first=0;first<n;first+=128) {
            int queries=n-first;if(queries>128)queries=128;
            xe_gpu_int_arg(att,5,first*16);xe_vision_launch(e,att,(queries*16+7)/8,1);
        }
        xe_vision_dense(e,attention,VW("attn_out.weight"),projected,n,1152,1152);
        xe_vision_norm(e,projected,VW("attn_post_norm.weight"),norm,n,1152);
        xe_vision_binary(e,4,hidden,norm,(int)wide);
        xe_vision_norm(e,hidden,VW("ln2.weight"),norm,n,1152);
        xe_vision_dense(e,norm,VW("ffn_gate.weight"),gate,n,1152,4304);
        xe_vision_dense(e,norm,VW("ffn_up.weight"),up,n,1152,4304);
        xe_vision_binary(e,5,gate,up,(int)ff);
        xe_vision_dense(e,gate,VW("ffn_down.weight"),projected,n,4304,1152);
        xe_vision_norm(e,projected,VW("ffn_post_norm.weight"),norm,n,1152);
        xe_vision_binary(e,4,hidden,norm,(int)wide);
        snprintf(name,sizeof name,"layer_out-%d",layer);xe_vision_dump(e,dump,name,hidden,wide);
#undef VW
    }
    ze_kernel_handle_t pool=xe_vision_kernel(e,6);
    xe_gpu_pointer_arg(pool,0,hidden);xe_gpu_pointer_arg(pool,1,xe_vision_weight(e,"v.std_bias",0));
    xe_gpu_pointer_arg(pool,2,xe_vision_weight(e,"v.std_scale",0));xe_gpu_pointer_arg(pool,3,norm);
    xe_gpu_int_arg(pool,4,cols);xe_gpu_int_arg(pool,5,im->rows);
    xe_vision_launch(e,pool,(im->rows*1152+127)/128,1);
    xe_vision_dump(e,dump,"std_scaled",norm,(size_t)im->rows*1152);
    xe_vision_norm(e,norm,NULL,projected,im->rows,1152);
    float *result=xe_vision_alloc_shared(e->owner,(size_t)im->rows*2816*4);
    xe_vision_dense(e,projected,"mm.input_projection.weight",result,im->rows,1152,2816);
    xe_vision_wait(e);
    for(size_t i=0;i<(size_t)im->rows*2816;i++)if(!isfinite(result[i]))xe_fatal("vision: nonfinite projected embedding");
    memcpy(output,result,(size_t)im->rows*2816*4);
    xe_vision_dump(e,dump,"embedding",result,(size_t)im->rows*2816);
    xe_vision_free_shared(e->owner,result);xe_vision_free_shared(e->owner,up);xe_vision_free_shared(e->owner,gate);
    xe_vision_free_shared(e->owner,projected);xe_vision_free_shared(e->owner,attention);
    xe_vision_free_shared(e->owner,v);xe_vision_free_shared(e->owner,k);xe_vision_free_shared(e->owner,q);
    xe_vision_free_shared(e->owner,norm);xe_vision_free_shared(e->owner,hidden);
    return 1;
}
