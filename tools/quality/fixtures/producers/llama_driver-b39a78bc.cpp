// Exact-token full-logit pinned baseline. No sampler or model-arithmetic changes.
#include "llama.h"
#include "ggml-backend.h"
#include <cstring>
#ifdef XE_QUALITY_VULKAN
#include <vulkan/vulkan.h>
#endif
#include <cmath>
#ifdef XE_QUALITY_CHUNK_COOLDOWN
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <thread>
#endif
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef XE_QUALITY_VULKAN
static bool full_offload = false;
static void capture_log(enum ggml_log_level, const char *text, void *) {
    fputs(text, stderr);
    const char *p = strstr(text, "offloaded ");
    int actual = 0, expected = 0;
    if (p && sscanf(p, "offloaded %d/%d layers to GPU", &actual, &expected) == 2)
        full_offload = expected > 0 && actual == expected;
}

static ggml_backend_dev_t select_vulkan() {
    auto device = ggml_backend_dev_by_name(XE_QUALITY_DEVICE);
    if (!device || strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "Vulkan") ||
        (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU &&
         ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_IGPU))
        throw std::runtime_error("explicit Vulkan GPU unavailable");
    // A software ICD can advertise ggml's GPU device type. Check Vulkan itself.
    VkInstanceCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    VkInstance instance;
    if (vkCreateInstance(&info, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("Vulkan loader initialization failed");
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(instance, &count, nullptr) != VK_SUCCESS || !count)
        throw std::runtime_error("no Vulkan physical devices");
    std::vector<VkPhysicalDevice> physical(count);
    if (vkEnumeratePhysicalDevices(instance, &count, physical.data()) != VK_SUCCESS)
        throw std::runtime_error("Vulkan enumeration failed");
    bool hardware = false;
    for (auto d : physical) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        if (!strcmp(props.deviceName, ggml_backend_dev_description(device)) &&
            (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
             props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)) {
            hardware = true;
            fprintf(stderr, "XE_QUALITY_VULKAN device=%s description=%s vendor_id=%u device_id=%u driver_version=%u api_version=%u physical_type=%u\n",
                    XE_QUALITY_DEVICE, props.deviceName, props.vendorID, props.deviceID,
                    props.driverVersion, props.apiVersion, props.deviceType);
        }
    }
    vkDestroyInstance(instance, nullptr);
    if (!hardware) throw std::runtime_error("selected Vulkan device is not a physical hardware GPU");
    auto backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) throw std::runtime_error("Vulkan GPU backend initialization failed");
    ggml_backend_free(backend);
    return device;
}

static void runtime_files() {
    std::ifstream maps("/proc/self/maps");
    if (!maps) throw std::runtime_error("Vulkan runtime mapping requires Linux procfs");
    std::string line;
    while (std::getline(maps, line)) {
        auto start = line.find('/');
        if (start != std::string::npos && line.find(".so", start) != std::string::npos)
            printf("runtime_file\t%s\n", line.substr(start).c_str());
    }
}
#endif

#ifdef XE_QUALITY_CHUNK_COOLDOWN
// Internal execution variant: synchronize and cool between original chunks.
// No changes to tokens, chunk schedule, batch construction, or model arithmetic.
static double monotonic_seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double max_hwmon_temp(const std::filesystem::path &root = "/sys/class/hwmon") {
    double maximum = -std::numeric_limits<double>::infinity();
    for (const auto &hwmon : std::filesystem::directory_iterator(root)) {
        if (!hwmon.is_directory()) continue;
        const auto dirname = hwmon.path().filename().string();
        if (dirname.rfind("hwmon", 0) != 0) continue;
        for (const auto &entry : std::filesystem::directory_iterator(hwmon.path())) {
            const auto name = entry.path().filename().string();
            if (name.rfind("temp", 0) != 0 || name.size() < 10 ||
                name.compare(name.size()-6, 6, "_input") != 0) continue;
            std::ifstream input(entry.path());
            if (!input) continue; // Same available-sensor scope as the outer guard.
            double raw;
            if (!(input >> raw) || !std::isfinite(raw))
                throw std::runtime_error("invalid/nonfinite hwmon temperature");
            const double temp = raw / 1000.0;
            if (temp > 0.0) maximum = std::max(maximum, temp);
        }
    }
    if (!std::isfinite(maximum)) throw std::runtime_error("no available finite hwmon temperature");
    return maximum;
}

// Pure timing state: injectable samples/timestamps make hysteresis reviewable
// independently of a model or GPU. Caller validates samples and enforces120s.
static bool cooldown_ready(double temp, double now, double started, double &cool_since) {
    if (!std::isfinite(temp) || temp > 95.0 || !std::isfinite(now) || now < started)
        throw std::runtime_error("invalid cooldown sample/time");
    if (temp <= 60.0) {
        if (cool_since < 0.0) cool_since = now;
    } else {
        cool_since = -1.0;
    }
    return now-started >= 5.0 && cool_since >= 0.0 && now-cool_since >= 5.0;
}

static void chunk_marker(const char *event, const std::string &key, size_t index,
                         int start, int tokens, size_t rows, double temp,
                         double boundary_elapsed = 0.0) {
    fprintf(stderr, "XE_QUALITY_CHUNK event=%s case=%s chunk=%zu start=%d tokens=%d rows=%zu max_temp=%.3f monotonic=%.6f boundary_elapsed=%.6f\n",
            event, key.c_str(), index, start, tokens, rows, temp,
            monotonic_seconds(), boundary_elapsed);
    fflush(stderr);
}

static void cool_before_chunk(const std::string &key, size_t index, int start,
                              int tokens, size_t rows) {
    const double started = monotonic_seconds();
    double cool_since = -1.0;
    while (true) {
        const double temp = max_hwmon_temp();
        const double now = monotonic_seconds();
        chunk_marker("cooldown", key, index, start, tokens, rows, temp, now-started);
        if (now-started >= 120.0) throw std::runtime_error("chunk cooldown exceeded120s boundary limit");
        if (cooldown_ready(temp, now, started, cool_since)) {
            chunk_marker("begin", key, index, start, tokens, rows, temp, now-started);
            return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

#endif

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
#ifdef XE_QUALITY_VULKAN
        if (argc == 2 && std::string(argv[1]) == "--probe") {
            llama_backend_init();
            select_vulkan();
            runtime_files();
            llama_backend_free();
            return 0;
        }
#endif
        if (argc != 6 || std::string(argv[5]) != "prefill") return 2;
        uint32_t endian = 1;
        if (*(unsigned char *)&endian != 1 || sizeof(float) != 4) return 2;
        int capacity = std::stoi(argv[4]);
        if (capacity < 64 || capacity > 262144) return 2;
        llama_backend_init();
        auto mp = llama_model_default_params();
#ifdef XE_QUALITY_VULKAN
        llama_log_set(capture_log, nullptr);
        ggml_backend_dev_t devices[] = {select_vulkan(), nullptr};
        mp.devices = devices;
        mp.n_gpu_layers = 999;
        mp.split_mode = LLAMA_SPLIT_MODE_NONE;
        mp.main_gpu = 0;
#else
        mp.n_gpu_layers = 0;
#endif
        mp.use_extra_bufts = true;
        auto *model = llama_model_load_from_file(argv[1], mp);
        if (!model) return 1;
#ifdef XE_QUALITY_VULKAN
        if (!full_offload) throw std::runtime_error("full Vulkan layer offload was not confirmed");
        fprintf(stderr, "XE_QUALITY_VULKAN dispatch=full-layer-offload flash_attn=OFF offload_kqv=1 op_offload=1\n");
#endif
        auto cp = llama_context_default_params();
        cp.n_ctx = capacity;
        cp.n_batch = 1024;
        cp.n_ubatch = 1024;
        cp.n_seq_max = 1;
        cp.n_threads = 8;
        cp.n_threads_batch = 8;
        cp.type_k = GGML_TYPE_F16;
        cp.type_v = GGML_TYPE_F16;
#ifdef XE_QUALITY_VULKAN
        cp.offload_kqv = true;
        cp.op_offload = true;
#else
        cp.offload_kqv = false;
        cp.op_offload = false;
#endif
        cp.swa_full = true;
        cp.kv_unified = true;
        cp.no_perf = true;
#ifdef XE_QUALITY_VULKAN
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
#else
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
#endif
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
#ifdef XE_QUALITY_CHUNK_COOLDOWN
            size_t completed_rows = 0;
            size_t chunk_index = 0;
#endif
            for (int n : chunks) {
#ifdef XE_QUALITY_CHUNK_COOLDOWN
                // Covers first chunk after context initialization/cache clear,
                // and every subsequent original chunk boundary.
                llama_synchronize(ctx);
                cool_before_chunk(cols[0], chunk_index, start, n, completed_rows);
#endif
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
#ifdef XE_QUALITY_CHUNK_COOLDOWN
                llama_synchronize(ctx);
#endif
                for (int i = 0; i < n; ++i) if (batch.logits[i]) {
                    const float *z = llama_get_logits_ith(ctx, i);
                    if (!z) return 1;
                    for (int v = 0; v < vocab; ++v) if (!std::isfinite(z[v])) return 1;
                    if (fwrite(z, sizeof(float), vocab, out) != (size_t)vocab) return 1;
#ifdef XE_QUALITY_CHUNK_COOLDOWN
                    ++completed_rows;
#endif
                }
#ifdef XE_QUALITY_CHUNK_COOLDOWN
                if (fflush(out)) return 1;
                const double end_temp = max_hwmon_temp();
                chunk_marker("end", cols[0], chunk_index, start, n, completed_rows, end_temp);
                if (end_temp > 95.0) throw std::runtime_error("chunk end exceeded hard95C limit");
#endif
                llama_batch_free(batch);
                start += n;
#ifdef XE_QUALITY_CHUNK_COOLDOWN
                ++chunk_index;
#endif
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
