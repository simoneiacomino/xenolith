// Exact-token full-logit CPU baseline. No sampler or model-arithmetic changes.
#include "llama.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::vector<int> integers(const std::string &s) {
    std::vector<int> values;
    std::stringstream stream(s);
    std::string item;
    while (std::getline(stream, item, ',')) {
        size_t used;
        int value = std::stoi(item, &used);
        if (used != item.size() || value < 0 || value > 262144)
            throw std::runtime_error("invalid integer input");
        values.push_back(value);
    }
    if (values.empty()) throw std::runtime_error("empty input field");
    return values;
}

int main(int argc, char **argv) {
    try {
        if (argc != 6 || std::string(argv[5]) != "prefill") return 2;
        uint32_t endian = 1;
        if (*(unsigned char *)&endian != 1 || sizeof(float) != 4) return 2;
        int capacity = std::stoi(argv[4]);
        if (capacity < 64 || capacity > 262144) return 2;
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        mp.use_extra_bufts = true;
        auto *model = llama_model_load_from_file(argv[1], mp);
        if (!model) return 1;
        auto cp = llama_context_default_params();
        cp.n_ctx = capacity;
        cp.n_batch = 1024;
        cp.n_ubatch = 1024;
        cp.n_seq_max = 1;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
        cp.offload_kqv = false;
        cp.op_offload = false;
        cp.swa_full = true;
        cp.kv_unified = true;
        cp.no_perf = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        auto *ctx = llama_init_from_model(model, cp);
        if (!ctx) return 1;
        int vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        if (vocab != 262144) return 2;
        std::ifstream input(argv[2]);
        if (!input) return 2;
        std::string line;
        int cases = 0;
        while (std::getline(input, line)) {
            std::stringstream fields(line);
            std::vector<std::string> cols;
            std::string col;
            while (std::getline(fields, col, '\t')) cols.push_back(col);
            if (cols.size() != 4 || cols[0].empty() || cols[0].size() > 96 ||
                cols[0].find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789-") != std::string::npos)
                return 2;
            auto ids = integers(cols[1]), pos = integers(cols[2]), chunks = integers(cols[3]);
            if (ids.size() > (size_t)capacity) return 2;
            for (int id : ids) if (id >= vocab) return 2;
            for (size_t i = 0; i < pos.size(); ++i)
                if (pos[i] >= (int)ids.size() || (i && pos[i] <= pos[i-1])) return 2;
            int total = 0;
            for (int n : chunks) {
                if (n < 1 || n > 1024 || n > (int)ids.size() - total) return 2;
                total += n;
            }
            if (total != (int)ids.size()) return 2;
            llama_memory_clear(llama_get_memory(ctx), true);
            const auto path = std::string(argv[3]) + "/" + cols[0] + ".bin";
            FILE *out = fopen(path.c_str(), "wbx");
            if (!out) return 2;
            int start = 0;
            size_t selected = 0;
            for (int n : chunks) {
                auto batch = llama_batch_init(n, 0, 1);
                batch.n_tokens = n;
                for (int i = 0; i < n; ++i) {
                    batch.token[i] = ids[start+i];
                    batch.pos[i] = start+i;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i] = selected < pos.size() && start+i == pos[selected];
                    if (batch.logits[i]) ++selected;
                }
                if (llama_decode(ctx, batch)) return 1;
                for (int i = 0; i < n; ++i) if (batch.logits[i]) {
                    const float *z = llama_get_logits_ith(ctx, i);
                    if (!z) return 1;
                    for (int v = 0; v < vocab; ++v) if (!std::isfinite(z[v])) return 1;
                    if (fwrite(z, sizeof(float), vocab, out) != (size_t)vocab) return 1;
                }
                llama_batch_free(batch);
                start += n;
            }
            if (fclose(out) || selected != pos.size()) return 2;
            ++cases;
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return cases ? 0 : 2;
    } catch (const std::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        return 2;
    }
}
