CC=gcc
ifneq ($(origin CTX),undefined)
$(error CTX is now a runtime option; use --ctx N when starting xenolith)
endif
BUILD_COMMIT=$(shell git rev-parse --short=9 HEAD 2>/dev/null || printf unknown)
BASE_CFLAGS=-O3 -march=native -std=c11 -Wall -Wextra -pthread
NUMERIC_SOURCES=xenolith.c xenolith.cl cpu_policy.c cpu_policy.h cpu_topology.c cpu_topology.h vision.c vision.h vision_encoder.c vision_encoder.h vision_internal.h vision.cl vendor/stb/stb_image.h
NUMERIC_SOURCE_HASH=$(shell sha256sum $(NUMERIC_SOURCES) | sha256sum | cut -d' ' -f1)
NUMERIC_CFLAGS_HASH=$(shell printf '%s' '$(BASE_CFLAGS)' | sha256sum | cut -d' ' -f1)
CFLAGS=$(BASE_CFLAGS) -DXE_BUILD_COMMIT='"$(BUILD_COMMIT)"' \
	-DXE_NUMERIC_SOURCE_HASH='"$(NUMERIC_SOURCE_HASH)"' \
	-DXE_NUMERIC_CFLAGS_HASH='"$(NUMERIC_CFLAGS_HASH)"'
LDLIBS=-lm -lze_loader

PREFIX ?= $(HOME)/.local
DESTDIR ?=
bindir ?= $(PREFIX)/bin
INSTALL ?= install

OCLOC_DEVICE?=xe-lp
GPU_SPV=xenolith_gpu.spv
GPU_OBJ=xenolith_gpu_spv.o
VISION_OBJS=vision.o vision_encoder.o
CPU_OBJS=cpu_policy.o cpu_topology.o

xenolith: main.o cli_config.o xenolith.o format.o $(VISION_OBJS) json.o profile.o kvstore.o conversation.o runtime.o serve.o $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -o $@ main.o cli_config.o xenolith.o format.o $(VISION_OBJS) json.o profile.o kvstore.o conversation.o runtime.o serve.o $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

# Install only the command. Native configuration belongs to the invoking user.
install: xenolith
	@test ! -L "$(DESTDIR)$(bindir)/xenolith" -a ! -d "$(DESTDIR)$(bindir)/xenolith" || { printf '%s\n' 'Refusing to replace a symlink or directory at the xenolith command path.' >&2; exit 1; }
	$(INSTALL) -d "$(DESTDIR)$(bindir)"
	$(INSTALL) -m 0755 xenolith "$(DESTDIR)$(bindir)/xenolith"

check-install: xenolith
	python3 tests/test_install.py ./xenolith

check-devices: xenolith
	CC="$(CC)" python3 tests/test_devices.py ./xenolith

.PHONY: install check-install check-devices

main.o: main.c xenolith.h format.h profile.h serve.h conversation.h kvstore.h cli_config.h
cli_config.o: cli_config.c cli_config.h json.h
xenolith.o: $(NUMERIC_SOURCES) xenolith.h xenolith_internal.h format.h
cpu_policy.o: cpu_policy.c cpu_policy.h
cpu_topology.o: cpu_topology.c cpu_topology.h cpu_policy.h
format.o: format.c format.h
json.o: json.c json.h
profile.o: profile.c profile.h xenolith.h format.h json.h
kvstore.o: kvstore.c kvstore.h xenolith.h xenolith_internal.h format.h
conversation.o: conversation.c conversation.h kvstore.h xenolith.h format.h
runtime.o: runtime.c runtime.h xenolith.h xenolith_internal.h kvstore.h conversation.h profile.h format.h json.h
serve.o: serve.c serve.h format.h runtime.h xenolith.h kvstore.h conversation.h profile.h json.h
main.o profile.o kvstore.o conversation.o runtime.o serve.o: vision.h

clean:
	rm -f tools/quality/native/xenolith_driver tools/quality/native/xe_tokenizer tools/quality/native/llama_tokenizer
	rm -f tests/test_cli_config tests/test_visual_cache tests/test_visual_replay tests/test_workers tests/test_cpu_policy tests/test_cpu_topology
	rm -f *.o xenolith tests/test_cache_policy tests/test_cache_model tests/test_tokenizer_model tests/test_kv tests/test_kv_model tests/test_decode tests/test_context_limits \
		tests/test_model_decode tests/test_fixture_decode bench/bench_decode \
		tests/test_prefill_projection tests/test_prefill_qkv tests/test_prefill_swa \
		tests/test_prefill_dense tests/test_prefill_layer tests/test_prefill_session \
		tests/test_prefill_session_drop tests/test_prefill_session_safe \
		tests/test_session_sync tests/test_format tests/test_snapshot \
		tests/test_snapshot_model \
		tests/test_kvstore tests/test_conversation \
		tests/test_conversation_model \
		tests/test_json tests/test_profile tests/test_inference tests/test_runtime \
		tests/test_runtime_model tests/test_serve tests/test_serve_model tests/test_gpu_alloc tests/test_prefill_host tests/test_cli_context tests/test_cli_workers \
		bench/bench_attention bench/bench_tg bench/bench_b3b bench/bench_b3b_8e \
		bench/bench_guard bench/compare_pp_tg_xe bench/compare_pp_tg_tokens \
		bench/bench_b3b.spv bench/bench_b3b_adlp.spv \
		bench/bench_prefill_gemm bench/bench_prefill_gemm_down \
		bench/bench_prefill_gemm.spv $(GPU_SPV) $(VISION_REFERENCE_TOOLS) tests/test_vision_input tests/test_vision_gpu tests/test_vision_session tests/test_text_batch_model tools/vision_prepare tests/test_sha256 tests/test_weights_identity

$(GPU_SPV): xenolith.cl vision.cl
	# Xe-LP is the SPIR-V feature baseline; native code is compiled at startup.
	ocloc compile -file $< -device $(OCLOC_DEVICE) -spv_only -output xenolith_gpu \
		-output_no_suffix -out_dir . -options '-cl-std=CL3.0' -q

$(GPU_OBJ): $(GPU_SPV) Makefile
	ld -r -b binary -z noexecstack -o $@ $<
	objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents $@

LLAMA_LDLIBS=-lllama -lggml -lggml-base

TEST_CFLAGS=-O2 -std=c11 -Wall -Wextra -pthread

HOST_UNIT_TESTS=tests/test_cpu_policy tests/test_cpu_topology tests/test_inference tests/test_format tests/test_sha256 tests/test_json tests/test_conversation tests/test_gpu_alloc tests/test_prefill_host tests/test_context_limits tests/test_visual_cache tests/test_visual_replay tests/test_vision_input
# These execute the engine's worker ISA check or VNNI arithmetic.
VNNI_TESTS=tests/test_workers tests/test_decode
GPU_TESTS=tests/test_kv tests/test_snapshot tests/test_kvstore tests/test_prefill_session tests/test_vision_gpu
PERSISTENCE_TESTS=tests/test_cache_model tests/test_snapshot_model tests/test_conversation_model tests/test_session_sync
PREFILL_TESTS=tests/test_prefill_projection tests/test_prefill_qkv tests/test_prefill_swa \
	tests/test_prefill_dense tests/test_prefill_layer tests/test_prefill_session
ENGINE_TESTS=tests/test_snapshot tests/test_snapshot_model tests/test_kvstore \
	tests/test_conversation_model tests/test_kv tests/test_kv_model tests/test_decode tests/test_model_decode \
	tests/test_fixture_decode $(PREFILL_TESTS) tests/test_prefill_session_drop tests/test_session_sync

$(ENGINE_TESTS) tests/test_cache_model tests/test_cache_policy tests/test_context_limits \
    tests/test_gpu_alloc tests/test_prefill_host tests/test_vision_gpu tests/test_vision_session \
    tests/test_visual_cache tests/test_visual_replay tests/test_weights_identity \
    bench/bench_decode bench/bench_attention bench/bench_tg bench/bench_b3b: \
    xenolith_internal.h format.h $(NUMERIC_SOURCES)
tests/test_prefill_session tests/test_prefill_session_drop \
    tests/test_kv_model tests/test_snapshot_model tests/test_session_sync tests/test_runtime_model \
    tests/test_conversation_model tests/test_serve_model: tests/test_context.h
tests/test_runtime tests/test_runtime_model tests/test_serve: xenolith.h conversation.h kvstore.h profile.h json.h
tests/test_serve: serve.h
tests/test_profile: json.h

MODEL?=
# Explicit inputs for the optional model/fixture regression runners below.
TEST_CONTEXT?=
VISION_PROMPT_IDS?=
VISION_EMBEDDING?=
MMPROJ?=
MMPROJ_VARIANT?=
MMPROJ_INVALID?=

tests/test_tokenizer_model: tests/test_tokenizer_model.c xenolith.o format.o $(VISION_OBJS) profile.o json.o xenolith.h profile.h $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_tokenizer_model.c xenolith.o format.o $(VISION_OBJS) profile.o json.o $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_gpu_alloc: tests/test_gpu_alloc.c xenolith.c xenolith.h xenolith_internal.h format.h
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -I. -o $@ $< \
		-Wl,--gc-sections -lm

tests/test_prefill_host: tests/test_prefill_host.c xenolith.c xenolith.h xenolith_internal.h format.h vision.c
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -I. -o $@ $< vision.c \
		-Wl,--gc-sections -lm

tests/test_cli_context: tests/test_cli_context.c xenolith.h xenolith
	$(CC) $(TEST_CFLAGS) -I. -o $@ $<

tests/test_cli_workers: tests/test_cli_workers.c xenolith
	$(CC) $(TEST_CFLAGS) -I. -o $@ $<

tests/test_context_limits: tests/test_context_limits.c cli_config.o main.c xenolith.c xenolith.h xenolith_internal.h runtime.c runtime.h serve.c serve.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< cli_config.o profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

check-context: tests/test_cli_context tests/test_cli_workers
	./tests/test_cli_context
	./tests/test_cli_workers

tests/test_format: tests/test_format.c format.o format.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_format.c format.o -pthread

# Compile without -march=native to exercise runtime SHA dispatch in host builds.
tests/test_sha256: tests/test_sha256.c format.c format.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ $<

tests/test_json: tests/test_json.c json.o json.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_json.c json.o -pthread -lm

tests/test_profile: tests/test_profile.c profile.o profile.h json.o format.o $(VISION_OBJS) xenolith.o xenolith.h $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_profile.c profile.o json.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_inference: tests/test_inference.c runtime.c runtime.h serve.c serve.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_inference.c profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_runtime: tests/test_runtime.c runtime.o runtime.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_runtime.c runtime.o profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_runtime_model: tests/test_runtime_model.c runtime.o runtime.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_runtime_model.c runtime.o profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_serve: tests/test_serve.c serve.o serve.h runtime.o runtime.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_serve.c serve.o runtime.o profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_serve_model: tests/test_serve_model.c json.o json.h xenolith
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_serve_model.c json.o -pthread -lm

tests/test_snapshot: tests/test_snapshot.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_snapshot.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_snapshot_model: tests/test_snapshot_model.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_snapshot_model.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_cache_model: tests/test_cache_model.c xenolith.c xenolith.h kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_cache_policy: tests/test_cache_policy.c tests/snapshot_compat_fixture.c xenolith.c xenolith.h kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_visual_cache: tests/test_visual_cache.c tests/snapshot_compat_fixture.c $(NUMERIC_SOURCES) xenolith_internal.h kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_kvstore: tests/test_kvstore.c xenolith.c xenolith.h kvstore.o kvstore.c kvstore.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_kvstore.c kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_conversation: tests/test_conversation.c conversation.o conversation.h kvstore.o kvstore.h xenolith.o xenolith.h format.o $(VISION_OBJS) format.h $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_conversation.c conversation.o kvstore.o xenolith.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_conversation_model: tests/test_conversation_model.c xenolith.c xenolith.h conversation.o conversation.h kvstore.o kvstore.h format.o $(VISION_OBJS) format.h $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_conversation_model.c conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_kv: tests/test_kv.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_kv.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_decode: tests/test_decode.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_decode.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_model_decode: tests/test_model_decode.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_model_decode.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_kv_model: tests/test_kv_model.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_kv_model.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_prefill_projection: tests/test_prefill_projection.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_projection.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_prefill_qkv: tests/test_prefill_qkv.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_qkv.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_prefill_swa: tests/test_prefill_swa.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_swa.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_prefill_dense: tests/test_prefill_dense.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_dense.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_prefill_layer: tests/test_prefill_layer.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_layer.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_prefill_session: tests/test_prefill_session.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_session.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_prefill_session_drop: tests/test_prefill_session.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -DXE_REPACK_DROP_SOURCE -I. -o $@ tests/test_prefill_session.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_session_sync: tests/test_session_sync.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_session_sync.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

bench/bench_decode: bench/bench_decode.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ bench/bench_decode.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

bench/bench_attention: bench/bench_attention.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ bench/bench_attention.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

bench/bench_tg: bench/bench_tg.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ bench/bench_tg.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

bench/bench_guard: bench/bench_guard.c
	$(CC) $(TEST_CFLAGS) -o $@ $<

bench/compare_pp_tg_xe: bench/compare_pp_tg.c xenolith.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ bench/compare_pp_tg.c xenolith.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

bench/compare_pp_tg_tokens: bench/compare_pp_tg_tokens.c
	$(CC) $(BASE_CFLAGS) -o $@ $<

bench/bench_b3b: bench/bench_b3b.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ bench/bench_b3b.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

# CPU maps are runtime settings; the former fixed E-core build was host-specific.
bench/bench_b3b_8e:
	@printf '%s\n' 'Use bench/bench_b3b with XE_CPU_LIST set to the verified E-core IDs on this machine.' >&2
	@exit 2

bench/bench_b3b.spv: bench/bench_b3b.cl
	ocloc compile -file $< -device xe-lp -spv_only -output bench_b3b \
		-output_no_suffix -out_dir bench -options '-cl-std=CL3.0' -q

bench-b3b: bench/bench_b3b bench/bench_b3b.spv

bench-b14: bench/bench_b3b bench/bench_b3b.spv

bench/bench_prefill_gemm: bench/bench_prefill_gemm.c
	$(CC) $(TEST_CFLAGS) -o $@ $< $(LDLIBS)

bench/bench_prefill_gemm_down: bench/bench_prefill_gemm.c
	$(CC) $(TEST_CFLAGS) -DBENCH_MOE_N=2816 -DBENCH_MOE_BLOCKS=22 \
		-o $@ $< $(LDLIBS)

bench/bench_prefill_gemm.spv: bench/bench_prefill_gemm.cl
	ocloc compile -file $< -device xe-lp -spv_only \
		-output bench_prefill_gemm -output_no_suffix -out_dir bench \
		-options '-cl-std=CL3.0' -q

bench-prefill-gemm: bench/bench_prefill_gemm bench/bench_prefill_gemm_down \
	bench/bench_prefill_gemm.spv

tests/test_fixture_decode: tests/test_fixture_decode.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_fixture_decode.c format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

check-kv: tests/test_kv
	./tests/test_kv

check-kv-edges: tests/test_kv
	./tests/test_kv edges

check-kv-load-bounds:
	CC='$(CC)' python3 tests/check_kv_load_bounds.py

check-kv check-kv-edges check-gpu: check-kv-load-bounds

check-inference: tests/test_inference
	./tests/test_inference

check-cache-large: tests/test_cache_policy
	./tests/test_cache_policy --large

check-snapshot-compat-host: $(GPU_OBJ) $(CPU_OBJS)
	python3 tests/check_snapshot_compat.py --current-only --cc '$(CC)' --base-cflags '$(BASE_CFLAGS)' --ocloc-device '$(OCLOC_DEVICE)'

# Needs the historical commit in local Git history, or LEGACY_SOURCE pointing
# to an unmodified export of it. No model or GPU device is needed.
check-snapshot-compat: $(GPU_OBJ) $(CPU_OBJS)
	python3 tests/check_snapshot_compat.py --cc '$(CC)' --base-cflags '$(BASE_CFLAGS)' --ocloc-device '$(OCLOC_DEVICE)' $(if $(LEGACY_SOURCE),--legacy-source '$(LEGACY_SOURCE)')

check-snapshot-compat-model: require-model $(GPU_OBJ) $(CPU_OBJS)
	python3 tests/check_snapshot_compat.py --cc '$(CC)' --base-cflags '$(BASE_CFLAGS)' --ocloc-device '$(OCLOC_DEVICE)' $(if $(LEGACY_SOURCE),--legacy-source '$(LEGACY_SOURCE)') --model '$(MODEL)'

check-vision-weights:
	CC='$(CC)' python3 tests/check_vision_weights.py

check-host-unit: check-vision-weights tests/test_cache_policy $(HOST_UNIT_TESTS) check-kv-load-bounds check-snapshot-compat-host
	./tests/test_cache_policy
	@set -e; for test in $(HOST_UNIT_TESTS); do ./$$test; done

check-vnni: $(VNNI_TESTS)
	@set -e; for test in $(VNNI_TESTS); do ./$$test; done

check-unit: check-host-unit check-vnni

# No GPU, model or VNNI execution. Compilation still uses the engine ISA flags.
check-host: check-host-unit check-context check-config check-install check-devices

check-gpu: $(GPU_TESTS)
	./tests/test_kv
	./tests/test_snapshot
	./tests/test_kvstore
	./tests/test_prefill_session route-reset
	./tests/test_vision_gpu

check: require-model tests/test_tokenizer_model tests/test_model_decode
	$(MAKE) check-unit
	$(MAKE) check-context
	$(MAKE) check-gpu
	./tests/test_model_decode "$(MODEL)"
	XENOLITH_MODEL="$(MODEL)" ./tests/test_tokenizer_model

check-persistence: require-model $(PERSISTENCE_TESTS)
	@set -e; for test in $(PERSISTENCE_TESTS); do ./$$test "$(MODEL)"; done

check-prefill: require-model $(PREFILL_TESTS) tests/test_fixture_decode
	@set -e; for test in $(PREFILL_TESTS); do ./$$test "$(MODEL)"; done
	./tests/test_fixture_decode "$(MODEL)"

check-all: require-model
	$(MAKE) check
	$(MAKE) check-persistence
	$(MAKE) check-prefill
	$(MAKE) check-runtime
	$(MAKE) check-serve

check-serve: require-model xenolith tests/test_serve tests/test_serve_model
	./tests/test_serve "$(MODEL)"
	./tests/test_serve_model "$(MODEL)"

check-runtime: require-model xenolith tests/test_json tests/test_profile tests/test_runtime \
		tests/test_runtime_model
	./tests/test_json
	./tests/test_profile "$(MODEL)"
	./tests/test_runtime "$(MODEL)"
	./tests/test_runtime_model "$(MODEL)"

# Optional CPU oracles, never linked into Xenolith. Build the two pinned llama.cpp
# checkouts separately first: encoder uses vision_reference_patch.py; decoder must
# retain the original vec.h. Neither source tree is patched by these targets.
LLAMA_VISION_DIR?=
LLAMA_VISION_BUILD?=$(LLAMA_VISION_DIR)/build-cpu
LLAMA_DECODE_DIR?=
LLAMA_DECODE_BUILD?=$(LLAMA_DECODE_DIR)/build-cpu
REFERENCE_CXXFLAGS?=-O2 -std=c++17 -Wall -Wextra -pthread
VISION_REFERENCE_TOOLS=tools/vision_reference tools/vision_decode_reference

# Rebuild on explicit request so a changed checkout/build path cannot reuse an
# oracle binary linked against a previous library directory.
.PHONY: $(VISION_REFERENCE_TOOLS) vision-reference-tools vision-reference-tools-clean
vision-reference-tools: $(VISION_REFERENCE_TOOLS)

tools/vision_reference: tools/vision_reference.cpp require-vision-reference
	$(CXX) $(REFERENCE_CXXFLAGS) -I"$(LLAMA_VISION_DIR)/include" \
		-I"$(LLAMA_VISION_DIR)/ggml/include" -I"$(LLAMA_VISION_DIR)/tools/mtmd" \
		-o $@ $< -L"$(LLAMA_VISION_BUILD)/bin" \
		-Wl,-rpath,"$(abspath $(LLAMA_VISION_BUILD)/bin)" -lmtmd $(LLAMA_LDLIBS)

tools/vision_decode_reference: tools/vision_decode_reference.cpp require-decode-reference
	$(CXX) $(REFERENCE_CXXFLAGS) -I"$(LLAMA_DECODE_DIR)/include" \
		-I"$(LLAMA_DECODE_DIR)/ggml/include" -o $@ $< \
		-L"$(LLAMA_DECODE_BUILD)/bin" -Wl,-rpath,"$(abspath $(LLAMA_DECODE_BUILD)/bin)" $(LLAMA_LDLIBS)

# Check the manifest revision, GELU variant, and CMake source/build association.
define require-reference-source
	@test -n "$(1)" -a -d "$(1)" || { printf '%s\n' 'Set $(3) to the appropriate pinned llama.cpp checkout.' >&2; exit 2; }
	@test -f "$(1)/ggml/src/ggml-cpu/vec.h" -a -f "$(2)/CMakeCache.txt" || { printf '%s\n' 'Reference sources and configured CMake build are required: $(1), $(2).' >&2; exit 2; }
	@git -C "$(1)" rev-parse --verify HEAD >/dev/null 2>&1 || { printf '%s\n' 'Reference source must be a Git checkout of the manifest revision.' >&2; exit 2; }
	@python3 -c 'import hashlib,json,pathlib,subprocess,sys; source=pathlib.Path(sys.argv[1]).resolve(); build=pathlib.Path(sys.argv[2]); refs=json.load(open("tests/vision/manifest.json"))["references"]; revision=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip(); revision==refs["llama_cpp_commit"] or sys.exit("reference checkout does not match manifest commit"); digest=hashlib.sha256((source/"ggml/src/ggml-cpu/vec.h").read_bytes()).hexdigest(); digest==refs["encoder_adapter"][sys.argv[3]] or sys.exit("reference vec.h has the wrong GELU variant"); cache=(build/"CMakeCache.txt").read_text(); ("CMAKE_HOME_DIRECTORY:INTERNAL="+str(source)) in cache.splitlines() or sys.exit("reference build does not belong to this source checkout")' "$(1)" "$(2)" "$(4)"
	@test -f "$(2)/bin/libllama.so" || { printf '%s\n' 'Build the reference CPU libraries in $(2) first.' >&2; exit 2; }
endef

require-vision-reference:
	$(call require-reference-source,$(LLAMA_VISION_DIR),$(LLAMA_VISION_BUILD),LLAMA_VISION_DIR,vec_h_adapted_sha256)
	@test -f "$(LLAMA_VISION_BUILD)/bin/libmtmd.so" || { printf '%s\n' 'Build the mtmd library in $(LLAMA_VISION_BUILD) first.' >&2; exit 2; }

require-decode-reference:
	$(call require-reference-source,$(LLAMA_DECODE_DIR),$(LLAMA_DECODE_BUILD),LLAMA_DECODE_DIR,vec_h_original_sha256)

vision-reference-tools-clean:
	rm -f $(VISION_REFERENCE_TOOLS)

.PHONY: require-vision-reference require-decode-reference

# These checks need explicit model/fixture inputs, so they are not in check-all.
# Validate inputs before building the integration binary, including with make -j.
check-vision-session-model: require-model require-test-context require-vision-session-inputs
	$(MAKE) tests/test_vision_session
	./tests/test_vision_session "$(MODEL)" "$(VISION_PROMPT_IDS)" "$(VISION_EMBEDDING)" "$(TEST_CONTEXT)"

check-text-batch-model: require-model require-test-context
	$(MAKE) tests/test_text_batch_model
	./tests/test_text_batch_model "$(MODEL)" "$(TEST_CONTEXT)"

# MMPROJ_VARIANT is a compatible tensor-body mutation; MMPROJ_INVALID has invalid metadata.
check-weights-identity: require-model require-weights-identity-inputs
	$(MAKE) tests/test_weights_identity
	./tests/test_weights_identity "$(MODEL)" "$(MMPROJ)" "$(MMPROJ_VARIANT)" "$(MMPROJ_INVALID)"

require-test-context:
	@case "$(TEST_CONTEXT)" in 65536|131072) ;; *) printf '%s\n' 'Set TEST_CONTEXT to 65536 or 131072.' >&2; exit 2;; esac

require-vision-session-inputs:
	@test -f "$(VISION_PROMPT_IDS)" || { printf '%s\n' 'Set VISION_PROMPT_IDS to a comma-separated prompt token ID file.' >&2; exit 2; }
	@test -f "$(VISION_EMBEDDING)" || { printf '%s\n' 'Set VISION_EMBEDDING to the projected embedding.f32 fixture.' >&2; exit 2; }

require-weights-identity-inputs:
	@test -f "$(MMPROJ)" || { printf '%s\n' 'Set MMPROJ to the pinned F16 projector GGUF.' >&2; exit 2; }
	@test -f "$(MMPROJ_VARIANT)" || { printf '%s\n' 'Set MMPROJ_VARIANT to a compatible projector with mutated tensor data.' >&2; exit 2; }
	@test -f "$(MMPROJ_INVALID)" || { printf '%s\n' 'Set MMPROJ_INVALID to a projector with invalid metadata.' >&2; exit 2; }

require-model:
	@test -f "$(MODEL)" || { printf '%s\n' 'Set MODEL to the path of the Gemma 4 GGUF file.' >&2; exit 2; }

.PHONY: check-cache-large require-model clean check check-inference check-host check-host-unit check-vnni check-unit check-gpu check-context check-persistence check-prefill check-all check-snapshot-compat-host check-snapshot-compat \
	check-snapshot-compat-model check-kv check-kv-edges check-kv-load-bounds check-runtime check-serve \
	bench-b3b bench-b14 bench-prefill-gemm check-vision check-vision-weights

.PHONY: check-vision-session-model check-text-batch-model check-weights-identity \
	require-test-context require-vision-session-inputs require-weights-identity-inputs

vision.o: vision.c vision.h vision_internal.h format.h vendor/stb/stb_image.h
	$(CC) $(CFLAGS) -c vision.c -o $@

vision_encoder.o: vision_encoder.c vision_encoder.h vision_internal.h vision.h format.h
	$(CC) $(CFLAGS) -c vision_encoder.c -o $@

tests/test_vision_input: tests/test_vision_input.c vision.o format.o
	$(CC) $(TEST_CFLAGS) -o $@ $^ -lm

tools/vision_prepare: tools/vision_prepare.c vision.o format.o
	$(CC) $(TEST_CFLAGS) -o $@ $^ -lm

tests/test_vision_gpu: tests/test_vision_gpu.c xenolith.c xenolith.h vision_encoder.c vision_internal.h $(VISION_OBJS) format.o $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< $(VISION_OBJS) format.o $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

check-vision: tests/test_vision_input tests/test_vision_gpu
	./tests/test_vision_input
	./tests/test_vision_gpu

tests/test_vision_session: tests/test_vision_session.c xenolith.c xenolith.h vision_encoder.c vision_internal.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_text_batch_model: tests/test_text_batch_model.c xenolith.o xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< xenolith.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_weights_identity: tests/test_weights_identity.c xenolith.c xenolith.h format.h $(NUMERIC_SOURCES) $(VISION_OBJS) format.o $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< $(VISION_OBJS) format.o $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

# Exercise visual media replay with deterministic encoding and real storage.
tests/test_visual_replay: tests/test_visual_replay.c runtime.c runtime.h xenolith.h xenolith_internal.h $(NUMERIC_SOURCES) profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

# Real CPU worker pool and scratch checks; no GPU/model is opened.
tests/test_workers: tests/test_workers.c xenolith.c xenolith.h xenolith_internal.h format.h $(NUMERIC_SOURCES) format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)
	$(CC) $(CFLAGS) -I. -o $@ $< format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS) $(LDLIBS)

tests/test_cpu_policy: tests/test_cpu_policy.c cpu_policy.c cpu_policy.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_cpu_policy.c cpu_policy.c

tests/test_cpu_topology: tests/test_cpu_topology.c cpu_topology.c cpu_topology.h cpu_policy.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_cpu_topology.c

check-workers: tests/test_cpu_policy tests/test_cpu_topology tests/test_workers tests/test_cli_workers
	./tests/test_cpu_policy
	./tests/test_cpu_topology
	./tests/test_workers
	./tests/test_cli_workers

.PHONY: check-workers bench/bench_b3b_8e

# Host-only configuration and CLI integration checks; no weights or GPU opened.
tests/test_cli_config: tests/test_cli_config.c cli_config.c cli_config.h json.c json.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_cli_config.c cli_config.c json.c -lm

check-config: tests/test_cli_config xenolith
	./tests/test_cli_config
	python3 tests/test_config_cli.py ./xenolith

.PHONY: check-config

# Engine build/package/release fixtures; no Node, model, GPU or Docker build.
check-engine-package:
	python3 -m unittest -v \
		installer.tests.test_build_profiles \
		installer.tests.test_build_tools \
		installer.tests.test_engine_version \
		installer.tests.test_engine_package \
		installer.tests.test_component_release \
		installer.tests.test_release_preflight

# Full installer/distribution suite, including engine packaging fixtures.
# Requires Node 26+ for existing XenoPi integration; no engine build or GPU.
check-installer:
	python3 -m unittest discover -s installer/tests -v

.PHONY: check-engine-package check-installer

# The driver links the same objects and flags as the current engine build.
# Use xe-quality-backend to force a rebuild and create an immutable receipt.
PYTHON ?= python3
QUALITY_NATIVE=tools/quality/native
QUALITY_OBJECTS=xenolith.o format.o $(VISION_OBJS) $(GPU_OBJ) $(CPU_OBJS)

$(QUALITY_NATIVE)/xenolith_driver: $(QUALITY_NATIVE)/xenolith_driver.c xenolith.h $(QUALITY_OBJECTS)
	$(CC) $(CFLAGS) -I. -o $@ $< $(QUALITY_OBJECTS) $(LDLIBS)

$(QUALITY_NATIVE)/xe_tokenizer: $(QUALITY_NATIVE)/xe_tokenizer.c xenolith.h $(QUALITY_OBJECTS)
	$(CC) $(CFLAGS) -I. -o $@ $< $(QUALITY_OBJECTS) $(LDLIBS)

quality-driver: $(QUALITY_NATIVE)/xenolith_driver
quality-tokenizer: $(QUALITY_NATIVE)/xe_tokenizer

# Independent tokenizer, against an explicitly supplied external llama.cpp build.
LLAMA_TOKENIZER_DIR ?=
LLAMA_TOKENIZER_BUILD ?=$(LLAMA_TOKENIZER_DIR)/build-cpu
quality-llama-tokenizer:
	@test -n "$(LLAMA_TOKENIZER_DIR)" -a -f "$(LLAMA_TOKENIZER_BUILD)/bin/libllama.so" || { printf '%s\n' 'Set LLAMA_TOKENIZER_DIR and LLAMA_TOKENIZER_BUILD to an explicit built llama.cpp checkout.' >&2; exit 2; }
	$(CXX) -O2 -std=c++17 -I"$(LLAMA_TOKENIZER_DIR)/include" -I"$(LLAMA_TOKENIZER_DIR)/ggml/include" -o $(QUALITY_NATIVE)/llama_tokenizer $(QUALITY_NATIVE)/llama_tokenizer.cpp -L"$(LLAMA_TOKENIZER_BUILD)/bin" -Wl,-rpath,"$(abspath $(LLAMA_TOKENIZER_BUILD)/bin)" $(LLAMA_LDLIBS)

quality-build-info:
	@printf '%s\n' 'CC=$(CC)' 'CFLAGS=$(CFLAGS)' 'LDLIBS=$(LDLIBS)' 'NUMERIC_SOURCE_HASH=$(NUMERIC_SOURCE_HASH)' 'NUMERIC_CFLAGS_HASH=$(NUMERIC_CFLAGS_HASH)'

check-quality-unit:
	$(PYTHON) -m unittest discover -s tests/quality -p 'test_*.py' -v

QUALITY_FIXTURES ?=tools/quality/fixtures/gemma4
QUALITY_BASELINE ?=
verify-quality-fixtures:
	$(PYTHON) -m xe_quality.fixtures verify --fixtures "$(QUALITY_FIXTURES)"

check-quality-fixtures: require-model
	@test -f "$(BACKEND_CONFIG)" -a -n "$(QUALITY_OUTPUT)" || { printf '%s\n' 'Set BACKEND_CONFIG and a new QUALITY_OUTPUT; see tools/quality/fixtures/README.md.' >&2; exit 2; }
	$(PYTHON) -m xe_quality.fixtures compare --fixtures "$(QUALITY_FIXTURES)" --model "$(MODEL)" --backend-config "$(BACKEND_CONFIG)" --output "$(QUALITY_OUTPUT)" $(if $(QUALITY_BASELINE),--baseline "$(QUALITY_BASELINE)")

# An explicit policy is required for acceptance. Without one the gate exits 3.
REFERENCE ?=
BACKEND_CONFIG ?=
QUALITY_OUTPUT ?=
POLICY ?=
check-quality: require-model
	@test -d "$(REFERENCE)" -a -f "$(BACKEND_CONFIG)" -a -n "$(QUALITY_OUTPUT)" || { printf '%s\n' 'Set REFERENCE, BACKEND_CONFIG and a new QUALITY_OUTPUT; see docs/testing.md.' >&2; exit 2; }
	$(PYTHON) -m xe_quality compare --archive "$(REFERENCE)" --model "$(MODEL)" --driver "$(QUALITY_NATIVE)/xenolith_driver" --backend-config "$(BACKEND_CONFIG)" --output "$(QUALITY_OUTPUT)" --timeout 1800 $(if $(POLICY),--policy "$(POLICY)")

CORPUS ?=
REFERENCE_CONFIG ?=
QUALIFICATION ?=
GOOGLE_SOURCE ?=
reference: require-model
	$(PYTHON) -m xe_quality build-reference --model "$(MODEL)" --google-source "$(GOOGLE_SOURCE)" --corpus "$(CORPUS)" --config "$(REFERENCE_CONFIG)" --qualification "$(QUALIFICATION)" --output "$(QUALITY_OUTPUT)"

tests/test_decode tests/test_kv tests/test_model_decode tests/test_prefill_layer: tests/support/scalar_ops.h

.PHONY: quality-driver quality-tokenizer quality-llama-tokenizer quality-build-info check-quality-unit check-quality reference
.PHONY: verify-quality-fixtures check-quality-fixtures
