#include "llama.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstring>

static bool decode(llama_context *ctx,const std::vector<llama_token> &tokens,
                    const std::vector<float> &embedding,int start,int rows) {
    size_t count=tokens.size();
    for(int phase=0;phase<3;phase++) {
        int first=phase==0?0:phase==1?start:start+rows;
        int n=phase==0?start:phase==1?rows:int(count)-start-rows;
        if(!n)continue;
        llama_batch b=llama_batch_init(n,phase==1?2816:0,1);b.n_tokens=n;
        for(int i=0;i<n;i++) {
            if(phase==1)std::memcpy(b.embd+size_t(i)*2816,embedding.data()+size_t(i)*2816,2816*4);
            else b.token[i]=tokens[first+i];
            b.pos[i]=first+i;b.n_seq_id[i]=1;b.seq_id[i][0]=0;
            b.logits[i]=size_t(first+i+1)==count;
        }
        llama_set_causal_attn(ctx,phase!=1);
        int result=llama_decode(ctx,b);llama_batch_free(b);
        if(result)return false;
    }
    llama_set_causal_attn(ctx,true);return true;
}

int main(int argc,char **argv) {
    if(argc!=8){std::cerr<<"usage: vision_decode_reference model ids embedding start rows ctx output-prefix\n";return 2;}
    std::ifstream f(argv[2]);std::string text((std::istreambuf_iterator<char>(f)),{});
    std::replace(text.begin(),text.end(),',',' ');std::istringstream in(text);
    std::vector<llama_token> tokens;llama_token t;while(in>>t)tokens.push_back(t);
    int start=std::atoi(argv[4]),rows=std::atoi(argv[5]),capacity=std::atoi(argv[6]);
    if(tokens.empty()||start<0||rows<1||rows>1120||start+rows>int(tokens.size())||capacity<int(tokens.size())+16)return 2;
    std::vector<float> embedding(size_t(rows)*2816);
    std::ifstream ef(argv[3],std::ios::binary);ef.read(reinterpret_cast<char *>(embedding.data()),embedding.size()*4);
    if(!ef||ef.peek()!=EOF)return 2;
    llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
    llama_model *model=llama_model_load_from_file(argv[1],mp);if(!model)return 1;
    auto cp=llama_context_default_params();cp.n_ctx=capacity;
    cp.n_batch=std::max(rows,512);cp.n_ubatch=cp.n_batch;
    // Keep the entire visual batch plus its local history, without allocating
    // all 128k positions in every SWA layer. Attention semantics are unchanged.
    cp.swa_full=false;
    cp.n_threads=6;cp.n_threads_batch=8;cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context *ctx=llama_init_from_model(model,cp);if(!ctx)return 1;
    if(!decode(ctx,tokens,embedding,start,rows))return 1;
    const llama_vocab *vocab=llama_model_get_vocab(model);int nv=llama_vocab_n_tokens(vocab);
    std::ofstream(std::string(argv[7])+".logits.f32",std::ios::binary).write(reinterpret_cast<const char *>(llama_get_logits(ctx)),nv*4);
    std::ofstream ids(std::string(argv[7])+".ids");
    for(int i=0;i<16;i++) {
        const float *logits=llama_get_logits(ctx);t=std::max_element(logits,logits+nv)-logits;
        ids<<t<<'\n';if(llama_vocab_is_eog(vocab,t))break;
        char piece[1024];int n=llama_token_to_piece(vocab,t,piece,sizeof piece,0,false);
        if(n>0)std::cout.write(piece,n);
        auto b=llama_batch_init(1,0,1);b.n_tokens=1;b.token[0]=t;b.pos[0]=tokens.size()+i;
        b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=1;
        int result=llama_decode(ctx,b);llama_batch_free(b);if(result)return 1;
    }
    std::cout<<'\n';llama_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
