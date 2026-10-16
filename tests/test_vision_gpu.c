#include "../xenolith.c"

static int attention_case(xe_engine *e,int rows,int previous,xe_attention_mode mode) {
    int keys=previous+rows;
    size_t q_count=(size_t)16*rows*256,k_count=(size_t)8*keys*256;
    float *q=xe_alloc(e,q_count*4,XE_MEM_SHARED),*out=xe_alloc(e,q_count*4,XE_MEM_SHARED);
    _Float16 *k=xe_alloc(e,k_count*2,XE_MEM_SHARED),*v=xe_alloc(e,k_count*2,XE_MEM_SHARED);
    memset(q,0,q_count*4);memset(k,0,k_count*2);
    for(size_t i=0;i<q_count;i++)out[i]=NAN;
    for(int head=0;head<8;head++)for(int key=0;key<keys;key++)for(int d=0;d<256;d++)
        v[((size_t)head*keys+key)*256+d]=(_Float16)((key%17)*0.125f+head*0.5f+d*0.0001f);
    xe_prefill_attention_online_append(e,q,k,v,out,rows,keys,256,8,previous,1024,mode);
    xe_gpu_wait(e);
    double *prefix=calloc((size_t)8*256*(keys+1),sizeof(double));
    if(!prefix)return 0;
    for(int head=0;head<8;head++)for(int d=0;d<256;d++) {
        double *p=prefix+((size_t)head*256+d)*(keys+1);
        for(int key=0;key<keys;key++)p[key+1]=p[key]+(float)v[((size_t)head*keys+key)*256+d];
    }
    double maximum=0;
    for(int head=0;head<16;head++)for(int row=0;row<rows;row++)for(int d=0;d<256;d++) {
        int first=previous+row-1023;if(first<0)first=0;
        const double *p=prefix+((size_t)(head/2)*256+d)*(keys+1);
        int end=mode==XE_ATTENTION_VISUAL?keys:previous+row+1;
        double sum=p[end]-p[first];
        double value=out[((size_t)head*rows+row)*256+d];
        double error=fabs(value-sum/(end-first));
        if(!isfinite(value)||!isfinite(error))maximum=INFINITY;
        else if(error>maximum)maximum=error;
    }
    free(prefix);
    int ok=maximum<1e-4;
    printf("%s attention: rows %d previous %d max %.3g %s\n",
           mode==XE_ATTENTION_VISUAL?"visual":"causal",rows,previous,maximum,ok?"PASS":"FAIL");
    xe_free(e,q,XE_MEM_SHARED);xe_free(e,out,XE_MEM_SHARED);xe_free(e,k,XE_MEM_SHARED);xe_free(e,v,XE_MEM_SHARED);
    return ok;
}

static int ring_case(xe_engine *e,int rows,int start) {
    size_t batch=(size_t)rows*8*256,ring=(size_t)1024*8*256;
    _Float16 *bk=xe_alloc(e,batch*2,XE_MEM_SHARED),*bv=xe_alloc(e,batch*2,XE_MEM_SHARED);
    _Float16 *rk=xe_alloc(e,ring*2,XE_MEM_SHARED),*rv=xe_alloc(e,ring*2,XE_MEM_SHARED);
    for(size_t i=0;i<batch;i++)bk[i]=bv[i]=(_Float16)((i/256%rows)%1024);
    for(size_t i=0;i<ring;i++)rk[i]=rv[i]=(_Float16)-1;
    xe_prefill_swa_commit_append(e,rk,rv,bk,bv,rows,start,256,8,1024);xe_gpu_wait(e);
    int ok=1,first=rows>1024?rows-1024:0;
    for(int head=0;head<8;head++)for(int r=first;r<rows;r++)for(int d=0;d<256;d++) {
        size_t dest=((size_t)head*1024+((start+r)&1023))*256+d;
        if(rk[dest]!=bk[((size_t)head*rows+r)*256+d]||rv[dest]!=bv[((size_t)head*rows+r)*256+d])ok=0;
    }
    printf("vision ring: rows %d start %d %s\n",rows,start,ok?"PASS":"FAIL");
    xe_free(e,bk,XE_MEM_SHARED);xe_free(e,bv,XE_MEM_SHARED);xe_free(e,rk,XE_MEM_SHARED);xe_free(e,rv,XE_MEM_SHARED);
    return ok;
}

int main(void) {
    xe_engine *e=calloc(1,sizeof *e);if(!e)return 2;
    e->context=65536;xe_worker_pool_init(e);xe_gpu_init(e);
    int ok=attention_case(e,17,7,XE_ATTENTION_CAUSAL)&&
           attention_case(e,17,7,XE_ATTENTION_VISUAL)&&
           attention_case(e,17,7,XE_ATTENTION_CAUSAL)&&
           attention_case(e,1120,1023,XE_ATTENTION_CAUSAL)&&
           attention_case(e,1120,1023,XE_ATTENTION_VISUAL)&&
           attention_case(e,1120,1023,XE_ATTENTION_CAUSAL)&&
           ring_case(e,512,900)&&ring_case(e,1024,23)&&ring_case(e,1120,977);
    xe_worker_pool_destroy(e);xe_gpu_destroy(e);xe_free(e,e->gelu_lut,XE_MEM_HOST);free(e);
    return ok?0:1;
}
