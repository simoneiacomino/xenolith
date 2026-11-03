#include "../xenolith.c"
#include "steering_fixture.h"

static void expected_ffn(float *out,const float *dense,const float *moe,const float *weights) {
    double ds=0,ms=0,cs=0;
    for(int j=0;j<XE_EMBD;j++){ds+=(double)dense[j]*dense[j];ms+=(double)moe[j]*moe[j];}
    ds=1/sqrt(ds/XE_EMBD+XE_RMS_EPS);ms=1/sqrt(ms/XE_EMBD+XE_RMS_EPS);
    for(int j=0;j<XE_EMBD;j++){out[j]=(float)(dense[j]*ds*weights[j]+moe[j]*ms*weights[XE_EMBD+j]);cs+=(double)out[j]*out[j];}
    cs=1/sqrt(cs/XE_EMBD+XE_RMS_EPS);
    for(int j=0;j<XE_EMBD;j++)out[j]=(float)(out[j]*cs*weights[2*XE_EMBD+j]);
}
int main(void){
    xe_engine *e=calloc(1,sizeof *e);assert(e);e->context=2048;e->model_content_hash_ready=1;
    xe_worker_pool_init(e);xe_gpu_init(e);
    char root[]="/tmp/xe-steering-ffn-XXXXXX";assert(mkdtemp(root));char pa[4096],pb[4096],pc[4096];
    snprintf(pa,sizeof pa,"%s/projection",root);snprintf(pb,sizeof pb,"%s/additive",root);snprintf(pc,sizeof pc,"%s/residual",root);
    xe_steering *a=steering_fixture_at(pa,XE_LAYERS,XE_EMBD,e->model_content_hash,"projection",XE_STEERING_FFN_SITE);
    xe_steering *b=steering_fixture_at(pb,XE_LAYERS,XE_EMBD,e->model_content_hash,"additive",XE_STEERING_FFN_SITE);
    xe_steering *c=steering_fixture(pc,XE_LAYERS,XE_EMBD,e->model_content_hash,"projection");
    xe_session s={0};s.engine=e;
    xe_steering_step steps[]={{0,a,0.5f},{3,c,0.5f},{8,NULL,0},{12,b,0.1f},{17,a,0},{20,a,-0.25f}};
    assert(xe_session_steering_schedule(&s,steps,6));
    float *weights=xe_alloc(e,(3*XE_EMBD+1)*sizeof(float),XE_MEM_SHARED);
    for(int j=0;j<3*XE_EMBD;j++)weights[j]=1+0.03f*sinf(j*0.01f);
    weights[3*XE_EMBD]=1.3f;
    xe_layer *l=&e->layers[0];l->post_ffw_norm1=weights;l->post_ffw_norm2=weights+XE_EMBD;
    l->post_ffw_norm=weights+2*XE_EMBD;l->layer_out_scale=weights+3*XE_EMBD;
    double maximum=0;
    const int counts[]={1,3,17,33,65,257};
    for(size_t z=0;z<sizeof counts/sizeof *counts;z++){
        int rows=counts[z];size_t n=(size_t)rows*XE_EMBD;
        float *raw=malloc(n*sizeof(float)),*expected=malloc(n*sizeof(float));assert(raw&&expected);
        float *dense=xe_alloc(e,(n+2)*sizeof(float),XE_MEM_SHARED);
        float *moe=xe_alloc(e,n*sizeof(float),XE_MEM_SHARED),*residual=xe_alloc(e,n*sizeof(float),XE_MEM_SHARED);
        float *hidden=xe_alloc(e,(n+2)*sizeof(float),XE_MEM_SHARED);
        dense[0]=hidden[0]=12345;dense[n+1]=hidden[n+1]=-12345;
        for(size_t i=0;i<n;i++){raw[i]=dense[i+1]=sinf((float)i*0.017f);moe[i]=cosf((float)i*0.013f);residual[i]=sinf((float)i*0.021f)*2;}
        xe_prefill_ffn_finish_append(e,l,dense+1,moe,residual,hidden+1,rows);xe_gpu_wait(e);
        memcpy(expected,hidden+1,n*sizeof(float));
        xe_prefill_workspace masked={0};masked.dense_down=dense+1;masked.moe_output=moe;
        masked.attention_output=residual;masked.hidden[1]=hidden+1;
        xe_steering_ffn_gpu(&s,1,&masked,rows,0);xe_gpu_wait(e);
        assert(!memcmp(expected,hidden+1,n*sizeof(float)));
        float row[XE_EMBD];
        for(int r=0;r<rows;r++){
            int selected=-1;for(int i=0;i<6;i++)if(steps[i].position<=r)selected=i;
            if(selected<0||steps[selected].strength==0||xe_steering_get_site(steps[selected].vector)!=XE_STEERING_FFN)continue;
            expected_ffn(row,raw+(size_t)r*XE_EMBD,moe+(size_t)r*XE_EMBD,weights);
            const float *v=xe_steering_row(steps[selected].vector,0);double scale=steps[selected].strength;
            if(xe_steering_get_mode(steps[selected].vector)==XE_STEERING_PROJECTION){double dot=0;for(int j=0;j<XE_EMBD;j++)dot+=(double)row[j]*v[j];scale=-scale*dot;}
            for(int j=0;j<XE_EMBD;j++){float changed=row[j]+(float)(scale*v[j]);expected[(size_t)r*XE_EMBD+j]=(changed+residual[(size_t)r*XE_EMBD+j])*weights[3*XE_EMBD];}
        }
        xe_prefill_workspace w={0};w.dense_down=dense+1;w.moe_output=moe;w.attention_output=residual;w.hidden[1]=hidden+1;
        xe_steering_ffn_gpu(&s,0,&w,rows,0);xe_gpu_wait(e);
        assert(dense[0]==12345&&dense[n+1]==-12345&&hidden[0]==12345&&hidden[n+1]==-12345);
        for(size_t i=0;i<n;i++){double err=fabs(hidden[i+1]-expected[i]);assert(isfinite(hidden[i+1]));if(err>maximum)maximum=err;}
        assert(maximum<3e-5);
        for(int r=0;r<rows;r++){
            int i=-1;for(int j=0;j<6;j++)if(steps[j].position<=r)i=j;
            if(i<0||steps[i].strength==0||xe_steering_get_site(steps[i].vector)!=XE_STEERING_FFN)
                assert(!memcmp(hidden+1+(size_t)r*XE_EMBD,expected+(size_t)r*XE_EMBD,XE_EMBD*sizeof(float)));
        }
        /* Capture must preserve the original fused hidden state byte for byte. */
        assert(xe_session_steering_schedule(&s,NULL,0));memcpy(dense+1,raw,n*sizeof(float));
        xe_prefill_ffn_finish_append(e,l,dense+1,moe,residual,hidden+1,rows);xe_gpu_wait(e);memcpy(expected,hidden+1,n*sizeof(float));
        assert(xe_session_capture_enable_at(&s,XE_STEERING_FFN));
        xe_steering_ffn_gpu(&s,0,&w,rows,0);xe_gpu_wait(e);
        assert(!memcmp(expected,hidden+1,n*sizeof(float)));
        expected_ffn(row,raw+(size_t)(rows-1)*XE_EMBD,moe+(size_t)(rows-1)*XE_EMBD,weights);
        for(int j=0;j<XE_EMBD;j++)assert(fabsf(row[j]-s.capture_rows[j])<3e-5f);
        xe_free(e,s.capture_rows,XE_MEM_SHARED);s.capture_rows=NULL;
        assert(xe_session_steering_schedule(&s,steps,6));
        free(raw);free(expected);xe_free(e,dense,XE_MEM_SHARED);xe_free(e,moe,XE_MEM_SHARED);xe_free(e,residual,XE_MEM_SHARED);xe_free(e,hidden,XE_MEM_SHARED);
    }
    xe_steering_clear(&s);xe_steering_release(a);xe_steering_release(b);xe_steering_release(c);
    steering_fixture_remove(pa);steering_fixture_remove(pb);steering_fixture_remove(pc);rmdir(root);
    xe_free(e,weights,XE_MEM_SHARED);xe_gpu_destroy(e);xe_worker_pool_destroy(e);free(e);
    printf("FFN steering: double reference, mixed sites/boundaries, zero rows, capture isolation and guards PASS; max %.3g\n",maximum);
}
