#include "../xenolith.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define REQUIRE(x) do {if(!(x)){fprintf(stderr,"text batch line %d: %s\n",__LINE__,#x);exit(1);}}while(0)

static int compare(xe_engine *e,int count,const int32_t *ids) {
    xe_session *s=xe_session_new(e);xe_tokens prefix={(int32_t *)ids,count,count};
    xe_session_sync(s,&prefix);int vocab=xe_vocab_size(e);
    float *baseline=malloc((size_t)vocab*4);REQUIRE(baseline);
    memcpy(baseline,xe_session_logits(s),(size_t)vocab*4);xe_session_free(s);
    s=xe_session_new(e);REQUIRE(xe_session_set_text_prefill(s,1024));xe_session_sync(s,&prefix);
    const float *candidate=xe_session_logits(s);double reference=0,error=0,dot=0,norm=0;
    int best_a=0,best_b=0;
    for(int i=0;i<vocab;i++) {
        REQUIRE(isfinite(baseline[i])&&isfinite(candidate[i]));
        double a=baseline[i],b=candidate[i];reference+=a*a;norm+=b*b;dot+=a*b;error+=(a-b)*(a-b);
        if(a>baseline[best_a])best_a=i;
        if(b>candidate[best_b])best_b=i;
    }
    double rms=sqrt(error/reference),cosine=dot/sqrt(reference*norm);
    int ok=rms<=0.005&&cosine>=0.99998&&best_a==best_b;
    printf("{\"context\":%d,\"tokens\":%d,\"relative_rms\":%.12g,\"cosine\":%.12g,\"argmax_512\":%d,\"argmax_1024\":%d,\"pass\":%s}\n",
           xe_context_size(e),count,rms,cosine,best_a,best_b,ok?"true":"false");fflush(stdout);
    free(baseline);xe_session_free(s);return ok;
}

int main(int argc,char **argv) {
    if(argc!=3)return 2;
    int context=atoi(argv[2]);REQUIRE(context==65536||context==131072);
    xe_engine *e=xe_engine_open_with_context(argv[1],context);
    const char *paragraph="Il sistema conserva la cronologia della conversazione e verifica i risultati. "
        "An image contains several objects, and attention connects the available information. "
        "For the following table, compute the totals accurately: January 17, February 23, March 31. "
        "def add(a, b): return a + b\n";
    char *text=calloc(1,100000);REQUIRE(text);
    for(int i=0;i<150;i++)strcat(text,paragraph);
    int32_t *ids=malloc(16384*4);REQUIRE(ids);ids[0]=xe_bos_id(e);
    int n=xe_encode_text_bounded(e,text,ids+1,16383);REQUIRE(n>=8191&&n<=16383);
    int ok=compare(e,2048,ids)&&compare(e,8192,ids);
    free(ids);free(text);xe_engine_close(e);return ok?0:1;
}
