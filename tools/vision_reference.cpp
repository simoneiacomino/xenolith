// Numerical oracle, built against the commit in tests/vision/manifest.json.
#include "clip.h"
#include "clip-impl.h"
#include "clip-model.h"
#include "ggml-backend.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>

static bool capture(ggml_tensor *t, bool ask, void *user) {
    const char *name = ggml_get_name(t);
    bool wanted = std::strstr(name,"pos_embd") || std::strstr(name,"layer_out") ||
                  std::strstr(name,"pooled") || std::strstr(name,"std_scaled") ||
                  std::strstr(name,"projected");
    if (ask) return wanted;
    if (!wanted) return true;
    auto path = std::filesystem::path(static_cast<const char *>(user)) / (std::string(name)+".bin");
    std::vector<char> bytes(ggml_nbytes(t));
    ggml_backend_tensor_get(t,bytes.data(),0,bytes.size());
    std::ofstream(path,std::ios::binary).write(bytes.data(),bytes.size());
    std::ofstream meta(path.string()+".json");
    meta << "{\"name\":\"" << name << "\",\"type\":" << int(t->type)
         << ",\"shape\":[" << t->ne[0] << ',' << t->ne[1] << ',' << t->ne[2] << ',' << t->ne[3]
         << "],\"bytes\":" << bytes.size() << "}\n";
    return true;
}

int main(int argc,char **argv) {
    if(argc!=4&&!(argc==5&&!std::strcmp(argv[4],"--no-dump"))){std::cerr<<"usage: vision_reference mmproj prepared.rgb output-dir [--no-dump]\n";return 2;}
    std::filesystem::create_directories(argv[3]);
    std::ifstream f(argv[2],std::ios::binary);
    uint32_t size[2]; f.read(reinterpret_cast<char *>(size),8);
    if(!f||!size[0]||!size[1]||size[0]%48||size[1]%48||uint64_t(size[0])*size[1]>1120*48*48){return 2;}
    std::vector<uint8_t> rgb(size_t(size[0])*size[1]*3);
    f.read(reinterpret_cast<char *>(rgb.data()),rgb.size());if(!f||f.peek()!=EOF)return 2;
    clip_image_u8 u8;u8.set_size({int(size[0]),int(size[1])},false);u8.cpy_buf(rgb);
    clip_image_f32 image;image.from_u8(u8);
    clip_context_params p{};p.use_gpu=false;p.warmup=false;
    p.flash_attn_type=CLIP_FLASH_ATTN_TYPE_DISABLED;p.image_min_tokens=1;p.image_max_tokens=1120;
    if(argc==4){p.cb_eval=capture;p.cb_eval_user_data=argv[3];}
    auto init=clip_init(argv[1],p);if(!init.ctx_v)return 1;
    // The locked GGUF omits clip.use_gelu; its HF config requires GELU-tanh.
    // Without this adapter, this llama.cpp revision falls back to QuickGELU.
    const_cast<clip_hparams *>(clip_get_hparams(init.ctx_v))->ffn_op=FFN_GELU;
    std::vector<float> output(size_t(clip_n_output_tokens(init.ctx_v,&image))*2816);
    auto start=std::chrono::steady_clock::now();
    bool ok=clip_image_encode(init.ctx_v,8,&image,output);
    std::cerr<<"reference encoder_ms="<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count()<<" threads=8 dumps="<<(argc==4)<<'\n';
    if(ok){
        auto path=std::filesystem::path(argv[3])/"embedding.f32";
        std::ofstream(path,std::ios::binary).write(reinterpret_cast<const char *>(output.data()),output.size()*4);
        std::cout<<"rows "<<output.size()/2816<<" width 2816\n";
    }
    clip_free(init.ctx_v);if(init.ctx_a)clip_free(init.ctx_a);
    return ok?0:1;
}
