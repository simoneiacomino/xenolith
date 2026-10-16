#include "../xenolith.c"
#define CHECK(x) do{if(!(x)){fprintf(stderr,"weights identity line %d: %s\n",__LINE__,#x);return 1;}}while(0)
int main(int argc,char **argv) {
    CHECK(argc==5);
    xe_engine *e=xe_engine_open_vocab(argv[1]);
    uint8_t original[32];memcpy(original,xe_engine_model_fingerprint(e),32);
    CHECK(xe_vision_weights_reference(original,0));
    /* A private COW page changes the mapped tensor without editing the file. */
    size_t page=(size_t)sysconf(_SC_PAGESIZE);CHECK(page>0);
    size_t last=e->map_len-1;uint8_t *byte=(uint8_t *)e->map+last;
    CHECK(!mprotect((uint8_t *)e->map+(last/page)*page,page,PROT_READ|PROT_WRITE));
    *byte^=1;e->model_content_hash_ready=0;
    const uint8_t *changed=xe_engine_model_fingerprint(e);
    CHECK(memcmp(changed,original,32)!=0);
    char error[256];CHECK(!xe_engine_enable_vision(e,"/missing-test-mmproj.gguf",error,sizeof error));
    CHECK(strstr(error,"cannot open mmproj"));
    *byte^=1;e->model_content_hash_ready=0;
    CHECK(!memcmp(original,xe_engine_model_fingerprint(e),32));xe_engine_close(e);
    xe_vision_model m;CHECK(xe_vision_model_open(&m,argv[2],error,sizeof error));
    CHECK(xe_vision_weights_reference(m.sha256,1));
    uint8_t projector[32];memcpy(projector,m.sha256,32);xe_vision_model_close(&m);
    int body_ok=xe_vision_model_open(&m,argv[3],error,sizeof error);
    CHECK(body_ok);CHECK(memcmp(projector,m.sha256,32));xe_vision_model_close(&m);
    CHECK(!xe_vision_model_open(&m,argv[4],error,sizeof error));
    puts("weights identity: full SHA-256, compatible tensor mutation and invalid metadata rejection PASS");return 0;
}
