#include "../xenolith.c"
#include "steering_fixture.h"

static void reference(float *h,const float *v,int width,int mode,float strength){
    if(!v||strength==0)return;
    double scale=strength;
    if(mode==2){double dot=0;for(int i=0;i<width;i++)dot+=(double)h[i]*v[i];scale=-strength*dot;}
    for(int i=0;i<width;i++)h[i]=(float)(h[i]+scale*v[i]);
}
int main(void){
    xe_engine *e=calloc(1,sizeof *e);assert(e);e->context=2048;e->model_content_hash_ready=1;
    xe_worker_pool_init(e);xe_gpu_init(e);
    char root[]="/tmp/xe-steering-gpu-XXXXXX";assert(mkdtemp(root));char pa[4096],pb[4096];
    snprintf(pa,sizeof pa,"%s/projection",root);snprintf(pb,sizeof pb,"%s/additive",root);
    xe_steering *a=steering_fixture(pa,XE_LAYERS,XE_EMBD,e->model_content_hash,"projection");
    xe_steering *b=steering_fixture(pb,XE_LAYERS,XE_EMBD,e->model_content_hash,"additive");
    xe_session s={0};s.engine=e;
    xe_steering_step steps[]={{0,a,-0.5f},{3,b,0.1f},{17,NULL,0},{31,a,1},{31,a,0.25f},{32,a,0}};
    assert(xe_session_steering_schedule(&s,steps,6));
    const int counts[]={1,3,17,33,65,512,1024};double maximum=0;
    for(size_t r=0;r<sizeof counts/sizeof *counts;r++)for(int start=0;start<=11;start+=11)for(int layer=0;layer<2;layer++){
        int rows=counts[r];size_t n=(size_t)rows*XE_EMBD;
        float *data=xe_alloc(e,(n+2)*sizeof *data,XE_MEM_SHARED),*expected=malloc(n*sizeof *expected);assert(expected);
        data[0]=12345;data[n+1]=-12345;
        for(size_t i=0;i<n;i++)data[i+1]=expected[i]=sinf((float)(i+1)*0.017f)*3;
        for(int row=0;row<rows;row++){
            int selected=-1;for(int i=0;i<6;i++)if(steps[i].position<=start+row)selected=i;
            if(selected>=0)reference(expected+(size_t)row*XE_EMBD,xe_steering_row(steps[selected].vector,layer),
                XE_EMBD,xe_steering_get_mode(steps[selected].vector),steps[selected].strength);
        }
        xe_steering_gpu(&s,layer,data+1,rows,start);xe_gpu_wait(e);
        assert(data[0]==12345&&data[n+1]==-12345);
        for(size_t i=0;i<n;i++){double error=fabs(data[i+1]-expected[i]);assert(isfinite(data[i+1]));if(error>maximum)maximum=error;}
        assert(maximum<3e-5);free(expected);xe_free(e,data,XE_MEM_SHARED);
    }
    uint8_t base[32],pending[32],effective[32];
    xe_steering_numeric(&s,0,base);xe_steering_numeric(&s,1,effective);assert(memcmp(base,effective,32));
    steps[5].strength=0.2f;assert(xe_session_steering_schedule(&s,steps,6));
    xe_steering_numeric(&s,1,pending);assert(!memcmp(pending,effective,32));
    xe_steering_clear(&s);xe_steering_release(a);xe_steering_release(b);
    steering_fixture_remove(pa);steering_fixture_remove(pb);rmdir(root);
    xe_gpu_destroy(e);xe_worker_pool_destroy(e);free(e);
    printf("steering GPU: 28 batch/mask/boundary cases, guard rows and pending-history identity PASS; max absolute error %.3g\n",maximum);
}
