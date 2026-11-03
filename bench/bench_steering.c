#define _POSIX_C_SOURCE 200809L
#include "../xenolith.h"
#include "../serve.h"
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Fixed input tokens isolate forward-pass cost from response length/sampling.
 * Run only when no other inference workload is active on the host. */
static double seconds(void) {
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}
int main(int argc,char **argv) {
    if(argc<4||argc>5){fprintf(stderr,"usage: %s MODEL MANIFEST STRENGTH [REPETITIONS]\n",argv[0]);return 2;}
    char *end;errno=0;float strength=strtof(argv[3],&end);
    if(errno||!*argv[3]||*end||!isfinite(strength))return 2;
    int repeats=3;
    if(argc==5){errno=0;long n=strtol(argv[4],&end,10);if(errno||!*argv[4]||*end||n<1||n>100)return 2;repeats=(int)n;}
    char holder[512];int lock=serve_lock(NULL,holder,sizeof holder);
    if(lock<0){fprintf(stderr,"benchmark cannot acquire engine lock: %s\n",holder);return 1;}
    double startup=seconds();xe_engine *e=xe_engine_open_cached(argv[1],1024);startup=seconds()-startup;
    xe_steering *vector=NULL;char error[256];
    if(!xe_steering_load(argv[2],xe_engine_layer_count(e),xe_engine_hidden_width(e),
        xe_engine_model_fingerprint(e),&vector,error,sizeof error)){
        fprintf(stderr,"%s\n",error);xe_engine_close(e);close(lock);return 1;
    }
    xe_session *s=xe_session_new(e);int32_t tokens[640];
    for(int i=0;i<640;i++)tokens[i]=2+(i*17)%101;
    /* One untimed warm-up per configuration. */
    for(int warm=0;warm<2;warm++){
        xe_session_reset(s);xe_session_steering_schedule(s,NULL,0);
        if(warm&&!xe_session_steering_set(s,vector,strength))return 1;
        xe_tokens p={tokens,512,640};xe_session_sync(s,&p);
        for(int i=512;i<640;i++){p.len=i+1;xe_session_sync(s,&p);}
    }
    for(int repeat=0;repeat<repeats;repeat++)for(int order=0;order<4;order++){
        int enabled=order==1||order==2;
        xe_session_reset(s);xe_session_steering_schedule(s,NULL,0);
        if(enabled&&!xe_session_steering_set(s,vector,strength))return 1;
        xe_tokens p={tokens,512,640};double began=seconds();xe_session_sync(s,&p);
        double prefill=seconds()-began;began=seconds();
        for(int i=512;i<640;i++){p.len=i+1;xe_session_sync(s,&p);}
        double decode=seconds()-began;
        if(prefill<=0||decode<=0||!isfinite(prefill)||!isfinite(decode))return 1;
        printf("{\"configuration\":\"%s\",\"repeat\":%d,\"order\":%d,\"strength\":%.9g,"
               "\"prefill_tokens\":512,\"prefill_seconds\":%.9g,\"decode_tokens\":128,\"decode_seconds\":%.9g,"
               "\"startup_seconds\":%.9g,\"artifact\":\"%s\"}\n",
               enabled?"steering":"off",repeat,order,(double)strength,prefill,decode,startup,xe_steering_id(vector));
        fflush(stdout);
    }
    xe_session_free(s);xe_steering_release(vector);xe_engine_close(e);close(lock);return 0;
}
