/* Gemma4v: valid rows are logical rows; rounded launches add no context. */
static inline float xe_vision_reduce8(float8 x) {
    float4 r=x.lo+x.hi;
    return (r.s0+r.s1)+(r.s2+r.s3);
}
__kernel void xe_vision_dense(__global const float *x, __global const half *w,
                              __global float *out, int rows, int width, int columns) {
    __local float a[16*32], b[16*32];
    int lid=get_local_id(0), r=get_group_id(0)*16+lid/8;
    int col=get_group_id(1)*16+(lid%8)*2;
    /* Independent stripes bound accumulation error for long F16 dots.
     * Reduce them as a balanced tree rather than a single long F32 chain. */
    float16 s0a=(float16)(0),s0b=(float16)(0),s1a=(float16)(0),s1b=(float16)(0);
    float tail0=0,tail1=0,comp0=0,comp1=0;
    for(int k0=0;k0<width;k0+=32) {
        for(int i=lid;i<16*32;i+=128) {
            int rr=get_group_id(0)*16+i/32, cc=get_group_id(1)*16+i/32, k=k0+i%32;
            a[i]=rr<rows&&k<width ? convert_float(convert_half_rte(x[(size_t)rr*width+k])):0;
            b[i]=cc<columns&&k<width ? convert_float(w[(size_t)cc*width+k]):0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if(k0+32<=width) {
            #pragma unroll
            for(int k=0;k<16;k++) {
                float v=a[(lid/8)*32+k],vh=a[(lid/8)*32+k+16];
                s0a[k]=fma(v,b[(lid%8)*2*32+k],s0a[k]);
                s1a[k]=fma(v,b[((lid%8)*2+1)*32+k],s1a[k]);
                s0b[k]=fma(vh,b[(lid%8)*2*32+k+16],s0b[k]);
                s1b[k]=fma(vh,b[((lid%8)*2+1)*32+k+16],s1b[k]);
            }
        } else {
            for(int k=0;k<width-k0;k++) {
                float v=a[(lid/8)*32+k];
                float y0=v*b[(lid%8)*2*32+k]-comp0,t0=tail0+y0;
                float y1=v*b[((lid%8)*2+1)*32+k]-comp1,t1=tail1+y1;
                comp0=(t0-tail0)-y0;comp1=(t1-tail1)-y1;tail0=t0;tail1=t1;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float s0=xe_vision_reduce8((s0a.lo+s0b.lo)+(s0a.hi+s0b.hi))+tail0;
    float s1=xe_vision_reduce8((s1a.lo+s1b.lo)+(s1a.hi+s1b.hi))+tail1;
    if(r<rows&&col<columns)out[(size_t)r*columns+col]=s0;
    if(r<rows&&col+1<columns)out[(size_t)r*columns+col+1]=s1;
}

__kernel void xe_vision_norm(__global const float *x, __global const float *weight,
                             __global float *out, int rows, int width) {
    int row=get_group_id(0), lane=get_local_id(0);
    if(row>=rows)return;
    float sum=0;
    for(int d=lane;d<width;d+=128){float v=x[(size_t)row*width+d];sum=fma(v,v,sum);}
    float scale=rsqrt(work_group_reduce_add(sum)/width+1e-6f);
    for(int d=lane;d<width;d+=128)
        out[(size_t)row*width+d]=x[(size_t)row*width+d]*scale*(weight?weight[d]:1.0f);
}

__kernel void xe_vision_head(__global float *x,__global const float *weight,
                             int rows,int patch_columns,int rotary) {
    __local float value[72];
    int task=get_group_id(0),d=get_local_id(0),row=task/16;
    if(row>=rows)return;
    float v=d<72?x[(size_t)task*72+d]:0;
    float scale=rsqrt(work_group_reduce_add(v*v)/72+1e-6f);
    if(d<72)value[d]=v*scale*(weight?weight[d]:1.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    if(d<72) {
        float result=value[d];
        if(rotary) {
            int axis=d/36,within=d%36,k=within%18;
            int pos=axis?row/patch_columns:row%patch_columns;
            float angle=pos*pow(100.0f,-(float)k/18);
            result=value[d]*cos(angle)+(within<18?-1.0f:1.0f)*value[axis*36+(within+18)%36]*sin(angle);
        }
        x[(size_t)task*72+d]=result;
    }
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void xe_vision_attention(__global const float *q,__global const float *k,
                                  __global const float *v,__global float *out,int rows,int first_task) {
    int task=first_task+get_group_id(0)*8+get_sub_group_id(),lane=get_sub_group_local_id();
    int row=task/16,head=task%16;if(row>=rows)return;
    float maximum=-INFINITY,denominator=0,acc[5]={0,0,0,0,0};
    for(int key0=0;key0<rows;key0+=8) {
        int count=min(8,rows-key0);float scores[8],block_max=-INFINITY;
        for(int j=0;j<count;j++) {
            float dot=0;
            for(int d=lane;d<72;d+=16)dot=fma(q[(size_t)task*72+d],k[((size_t)(key0+j)*16+head)*72+d],dot);
            scores[j]=sub_group_reduce_add(dot);block_max=fmax(block_max,scores[j]);
        }
        float next=fmax(maximum,block_max),alpha=maximum==-INFINITY?0:exp(maximum-next);
        denominator*=alpha;for(int d=0;d<5;d++)acc[d]*=alpha;
        for(int j=0;j<count;j++) {
            float beta=exp(scores[j]-next);denominator+=beta;
            for(int d=lane;d<72;d+=16)acc[d/16]+=beta*v[((size_t)(key0+j)*16+head)*72+d];
        }
        maximum=next;
    }
    for(int d=lane;d<72;d+=16)out[(size_t)task*72+d]=acc[d/16]/denominator;
}

__kernel void xe_vision_add(__global float *residual,__global const float *x,int elements) {
    size_t i=get_global_id(0);if(i<(size_t)elements)residual[i]+=x[i];
}
__kernel void xe_vision_geglu(__global float *gate,__global const float *up,int elements) {
    size_t i=get_global_id(0);if(i>=(size_t)elements)return;
    float v=gate[i];gate[i]=0.5f*v*(1+tanh(0.7978845608028654f*(v+0.044715f*v*v*v)))*up[i];
}
__kernel void xe_vision_pool(__global const float *x,__global const float *bias,
                            __global const float *scale,__global float *out,int columns,int rows) {
    size_t i=get_global_id(0);if(i>=(size_t)rows*1152)return;
    int row=i/1152,d=i%1152,px=(row%(columns/3))*3,py=(row/(columns/3))*3;
    float sum=0;for(int y=0;y<3;y++)for(int xx=0;xx<3;xx++)sum+=x[((size_t)(py+y)*columns+px+xx)*1152+d];
    out[i]=(sum/9*33.94112549695428f-bias[d])*scale[d];
}
