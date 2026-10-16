CC=gcc
ifneq ($(origin CTX),undefined)
$(error CTX is now a runtime option; use --ctx N when starting xenolith)
endif
BUILD_COMMIT=$(shell git rev-parse --short=9 HEAD 2>/dev/null || printf unknown)
BASE_CFLAGS=-O3 -march=native -std=c11 -Wall -Wextra -pthread
NUMERIC_SOURCES=xenolith.c xenolith.cl vision.c vision.h vision_encoder.c vision_encoder.h vision_internal.h vision.cl vendor/stb/stb_image.h
NUMERIC_SOURCE_HASH=$(shell sha256sum $(NUMERIC_SOURCES) | sha256sum | cut -d' ' -f1)
NUMERIC_CFLAGS_HASH=$(shell printf '%s' '$(BASE_CFLAGS)' | sha256sum | cut -d' ' -f1)
CFLAGS=$(BASE_CFLAGS) -DXE_BUILD_COMMIT='"$(BUILD_COMMIT)"' \
	-DXE_NUMERIC_SOURCE_HASH='"$(NUMERIC_SOURCE_HASH)"' \
	-DXE_NUMERIC_CFLAGS_HASH='"$(NUMERIC_CFLAGS_HASH)"'
LDLIBS=-lm -lze_loader

GPU_SPV=xenolith_gpu.spv
GPU_OBJ=xenolith_gpu_spv.o
VISION_OBJS=vision.o vision_encoder.o

xenolith: main.o xenolith.o format.o $(VISION_OBJS) json.o profile.o kvstore.o conversation.o runtime.o serve.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -o $@ main.o xenolith.o format.o $(VISION_OBJS) json.o profile.o kvstore.o conversation.o runtime.o serve.o $(GPU_OBJ) $(LDLIBS)

main.o: main.c xenolith.h format.h profile.h serve.h conversation.h kvstore.h
xenolith.o: $(NUMERIC_SOURCES) xenolith.h xenolith_internal.h format.h
format.o: format.c format.h
json.o: json.c json.h
profile.o: profile.c profile.h xenolith.h format.h json.h
kvstore.o: kvstore.c kvstore.h xenolith.h xenolith_internal.h format.h
conversation.o: conversation.c conversation.h kvstore.h xenolith.h format.h
runtime.o: runtime.c runtime.h xenolith.h xenolith_internal.h kvstore.h conversation.h profile.h format.h json.h
serve.o: serve.c serve.h format.h runtime.h xenolith.h kvstore.h conversation.h profile.h json.h
main.o profile.o kvstore.o conversation.o runtime.o serve.o: vision.h

clean:
	rm -f tests/test_visual_cache tests/test_visual_replay
	rm -f *.o xenolith tests/test_cache_policy tests/test_cache_model tests/certify tests/test_kv tests/test_kv_model tests/test_decode tests/test_context_limits \
		tests/test_model_decode tests/test_fixture_decode bench/bench_decode \
		tests/test_prefill_projection tests/test_prefill_qkv tests/test_prefill_swa \
		tests/test_prefill_dense tests/test_prefill_layer tests/test_prefill_session \
		tests/test_prefill_session_drop tests/test_prefill_session_safe \
		tests/test_session_sync tests/test_format tests/test_snapshot \
		tests/test_snapshot_model \
		tests/test_kvstore tests/test_conversation \
		tests/test_conversation_model \
		tests/test_json tests/test_profile tests/test_inference tests/test_runtime \
		tests/test_runtime_model tests/test_serve tests/test_serve_model tests/test_gpu_alloc tests/test_prefill_host tests/test_cli_context \
		bench/bench_attention bench/bench_tg bench/bench_b3b bench/bench_b3b_8e \
		bench/bench_guard bench/compare_pp_tg_xe bench/compare_pp_tg_tokens \
		bench/bench_b3b.spv bench/bench_b3b_adlp.spv \
		bench/bench_prefill_gemm bench/bench_prefill_gemm_down \
		bench/bench_prefill_gemm.spv $(GPU_SPV) $(TEST_TOOLS) $(VISION_REFERENCE_TOOLS) tests/test_vision_input tests/test_vision_gpu tests/test_vision_session tests/test_text_batch_model tools/vision_prepare tests/test_sha256 tests/test_weights_identity

$(GPU_SPV): xenolith.cl vision.cl
	# Xe-LP is the SPIR-V feature baseline; native code is compiled at startup.
	ocloc compile -file $< -device xe-lp -spv_only -output xenolith_gpu \
		-output_no_suffix -out_dir . -options '-cl-std=CL3.0' -q

$(GPU_OBJ): $(GPU_SPV)
	ld -r -b binary -o $@ $<
	objcopy --rename-section .data=.rodata,alloc,load,readonly,data,contents $@

LLAMA_DIR=../llama.cpp
LLAMA_BUILD=$(LLAMA_DIR)/build-cpu
LLAMA_LIBDIR=$(LLAMA_BUILD)/bin
LLAMA_CFLAGS=-I$(LLAMA_DIR)/include -I$(LLAMA_DIR)/ggml/include
LLAMA_LDFLAGS=-L$(LLAMA_LIBDIR) -Wl,-rpath,$(abspath $(LLAMA_LIBDIR))
LLAMA_LDLIBS=-lllama -lggml -lggml-base

TEST_CFLAGS=-O2 -std=c11 -Wall -Wextra -pthread

UNIT_TESTS=tests/test_inference tests/test_format tests/test_sha256 tests/test_json tests/test_decode tests/test_conversation tests/test_gpu_alloc tests/test_prefill_host tests/test_context_limits tests/test_visual_cache tests/test_visual_replay tests/test_vision_input
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
    bench/bench_decode bench/bench_attention bench/bench_tg bench/bench_b3b bench/bench_b3b_8e: \
    xenolith_internal.h format.h $(NUMERIC_SOURCES)
tests/test_prefill_session tests/test_prefill_session_drop \
    tests/test_kv_model tests/test_snapshot_model tests/test_session_sync tests/test_runtime_model \
    tests/test_conversation_model tests/test_serve_model: tests/test_context.h
tests/test_runtime tests/test_runtime_model tests/test_serve: xenolith.h conversation.h kvstore.h profile.h json.h
tests/test_serve: serve.h
tests/test_profile: json.h

VOCAB_DIR=$(LLAMA_DIR)/models
MODEL?=
# Explicit inputs for the optional model/fixture regression runners below.
TEST_CONTEXT?=
VISION_PROMPT_IDS?=
VISION_EMBEDDING?=
MMPROJ?=
MMPROJ_VARIANT?=
MMPROJ_INVALID?=
GOLDEN_PROMPTS=short long mixed ws nl nlonly json
ORACLE_PROMPTS=short long

tests/certify: tests/certify.c xenolith.o format.o $(VISION_OBJS) profile.o json.o xenolith.h profile.h $(GPU_OBJ)
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/certify.c xenolith.o format.o $(VISION_OBJS) profile.o json.o $(GPU_OBJ) $(LDLIBS)

tests/test_gpu_alloc: tests/test_gpu_alloc.c xenolith.c xenolith.h xenolith_internal.h format.h
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -I. -o $@ $< \
		-Wl,--gc-sections -lm

tests/test_prefill_host: tests/test_prefill_host.c xenolith.c xenolith.h xenolith_internal.h format.h vision.c
	$(CC) $(CFLAGS) -ffunction-sections -fdata-sections -I. -o $@ $< vision.c \
		-Wl,--gc-sections -lm

tests/test_cli_context: tests/test_cli_context.c xenolith.h xenolith
	$(CC) $(TEST_CFLAGS) -I. -o $@ $<

tests/test_context_limits: tests/test_context_limits.c main.c xenolith.c xenolith.h xenolith_internal.h runtime.c runtime.h serve.c serve.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ $< profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

check-context: tests/test_cli_context
	./tests/test_cli_context

tests/test_format: tests/test_format.c format.o format.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_format.c format.o -pthread

# Compile without -march=native to exercise runtime SHA dispatch in host builds.
tests/test_sha256: tests/test_sha256.c format.c format.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ $<

tests/test_json: tests/test_json.c json.o json.h
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_json.c json.o -pthread -lm

tests/test_profile: tests/test_profile.c profile.o profile.h json.o format.o $(VISION_OBJS) xenolith.o xenolith.h $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_profile.c profile.o json.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(LDLIBS)

tests/test_inference: tests/test_inference.c runtime.c runtime.h serve.c serve.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_inference.c profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(LDLIBS)

tests/test_runtime: tests/test_runtime.c runtime.o runtime.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_runtime.c runtime.o profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(LDLIBS)

tests/test_runtime_model: tests/test_runtime_model.c runtime.o runtime.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_runtime_model.c runtime.o profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(LDLIBS)

tests/test_serve: tests/test_serve.c serve.o serve.h runtime.o runtime.h profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_serve.c serve.o runtime.o profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) xenolith.o $(GPU_OBJ) $(LDLIBS)

tests/test_serve_model: tests/test_serve_model.c json.o json.h xenolith
	$(CC) $(TEST_CFLAGS) -I. -o $@ tests/test_serve_model.c json.o -pthread -lm

tests/test_snapshot: tests/test_snapshot.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_snapshot.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_snapshot_model: tests/test_snapshot_model.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_snapshot_model.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_cache_model: tests/test_cache_model.c xenolith.c xenolith.h kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ $< kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_cache_policy: tests/test_cache_policy.c tests/snapshot_compat_fixture.c xenolith.c xenolith.h kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ $< kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_visual_cache: tests/test_visual_cache.c tests/snapshot_compat_fixture.c $(NUMERIC_SOURCES) xenolith_internal.h kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ $< kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_kvstore: tests/test_kvstore.c xenolith.c xenolith.h kvstore.o kvstore.c kvstore.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_kvstore.c kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_conversation: tests/test_conversation.c conversation.o conversation.h kvstore.o kvstore.h xenolith.o xenolith.h format.o $(VISION_OBJS) format.h $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_conversation.c conversation.o kvstore.o xenolith.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_conversation_model: tests/test_conversation_model.c xenolith.c xenolith.h conversation.o conversation.h kvstore.o kvstore.h format.o $(VISION_OBJS) format.h $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_conversation_model.c conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_kv: tests/test_kv.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_kv.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_decode: tests/test_decode.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_decode.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_model_decode: tests/test_model_decode.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_model_decode.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_kv_model: tests/test_kv_model.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_kv_model.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_projection: tests/test_prefill_projection.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_projection.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_qkv: tests/test_prefill_qkv.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_qkv.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_swa: tests/test_prefill_swa.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_swa.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_dense: tests/test_prefill_dense.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_dense.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_layer: tests/test_prefill_layer.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_layer.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_session: tests/test_prefill_session.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_prefill_session.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_prefill_session_drop: tests/test_prefill_session.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -DXE_REPACK_DROP_SOURCE -I. -o $@ tests/test_prefill_session.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_session_sync: tests/test_session_sync.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_session_sync.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

bench/bench_decode: bench/bench_decode.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ bench/bench_decode.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

bench/bench_attention: bench/bench_attention.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ bench/bench_attention.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

bench/bench_tg: bench/bench_tg.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ bench/bench_tg.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

bench/bench_guard: bench/bench_guard.c
	$(CC) $(TEST_CFLAGS) -o $@ $<

bench/compare_pp_tg_xe: bench/compare_pp_tg.c xenolith.o format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ bench/compare_pp_tg.c xenolith.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

bench/compare_pp_tg_tokens: bench/compare_pp_tg_tokens.c
	$(CC) $(BASE_CFLAGS) -o $@ $<

bench/bench_b3b: bench/bench_b3b.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ bench/bench_b3b.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

bench/bench_b3b_8e: bench/bench_b3b.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -DXE_WORKERS=8 -DXE_WORKER_ECORES -I. \
		-o $@ bench/bench_b3b.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

bench/bench_b3b.spv: bench/bench_b3b.cl
	ocloc compile -file $< -device xe-lp -spv_only -output bench_b3b \
		-output_no_suffix -out_dir bench -options '-cl-std=CL3.0' -q

bench-b3b: bench/bench_b3b bench/bench_b3b.spv

bench-b14: bench/bench_b3b bench/bench_b3b_8e bench/bench_b3b.spv

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

tests/test_fixture_decode: tests/test_fixture_decode.c xenolith.c xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ tests/test_fixture_decode.c format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

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

check-snapshot-compat-host: $(GPU_OBJ)
	python3 tests/check_snapshot_compat.py --current-only --cc '$(CC)' --base-cflags '$(BASE_CFLAGS)'

# Needs the historical commit in local Git history, or LEGACY_SOURCE pointing
# to an unmodified export of it. No model or GPU device is needed.
check-snapshot-compat: $(GPU_OBJ)
	python3 tests/check_snapshot_compat.py --cc '$(CC)' --base-cflags '$(BASE_CFLAGS)' $(if $(LEGACY_SOURCE),--legacy-source '$(LEGACY_SOURCE)')

check-snapshot-compat-model: require-model $(GPU_OBJ)
	python3 tests/check_snapshot_compat.py --cc '$(CC)' --base-cflags '$(BASE_CFLAGS)' $(if $(LEGACY_SOURCE),--legacy-source '$(LEGACY_SOURCE)') --model '$(MODEL)'

check-vision-weights:
	CC='$(CC)' python3 tests/check_vision_weights.py

check-unit: check-vision-weights tests/test_cache_policy $(UNIT_TESTS) check-kv-load-bounds check-snapshot-compat-host
	./tests/test_cache_policy
	@set -e; for test in $(UNIT_TESTS); do ./$$test; done

check-gpu: $(GPU_TESTS)
	./tests/test_kv
	./tests/test_snapshot
	./tests/test_kvstore
	./tests/test_prefill_session route-reset
	./tests/test_vision_gpu

check: require-model tests/certify tests/test_model_decode
	$(MAKE) check-unit
	$(MAKE) check-context
	$(MAKE) check-gpu
	./tests/test_model_decode "$(MODEL)"
	XENOLITH_MODEL="$(MODEL)" ./tests/certify

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

golden: require-model test-tools
	mkdir -p tests/golden tests/fixtures
	@set -e; stage=$$(mktemp -d tests/.golden.XXXXXX); \
	trap 'rm -f "$$stage"/*; rmdir "$$stage"' EXIT; \
	trap 'exit 1' HUP INT TERM; \
	for p in $(GOLDEN_PROMPTS); do \
		tools/tokenize_prompt "$(MODEL)" tests/prompts/$$p.txt > "$$stage/$$p.ids"; \
	done; \
	for p in $(ORACLE_PROMPTS); do \
		ids=$$(tools/tokenize_prompt "$(MODEL)" tests/prompts/$$p.txt); \
		tools/dump_logits "$(MODEL)" "$$ids" "$$stage/$$p.logits"; \
	done; \
	cp "$(VOCAB_DIR)/ggml-vocab-gemma-4.gguf.inp" "$$stage/ggml-vocab-gemma-4.gguf.inp"; \
	cp "$(VOCAB_DIR)/ggml-vocab-gemma-4.gguf.out" "$$stage/ggml-vocab-gemma-4.gguf.out"; \
	for p in $(GOLDEN_PROMPTS); do mv "$$stage/$$p.ids" tests/golden/$$p.ids; done; \
	for p in $(ORACLE_PROMPTS); do mv "$$stage/$$p.logits" tests/golden/$$p.logits; done; \
	mv "$$stage/ggml-vocab-gemma-4.gguf.inp" tests/fixtures/ggml-vocab-gemma-4.gguf.inp; \
	mv "$$stage/ggml-vocab-gemma-4.gguf.out" tests/fixtures/ggml-vocab-gemma-4.gguf.out

TEST_TOOLS=tools/dump_logits tools/tokenize_prompt tools/dump_layers tools/compare_logits

test-tools: $(TEST_TOOLS)

tools/dump_logits: tools/dump_logits.c
	$(CC) $(TEST_CFLAGS) $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS) $(LLAMA_LDLIBS) $(LDLIBS)

tools/dump_layers: tools/dump_layers.c
	$(CC) $(TEST_CFLAGS) $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS) $(LLAMA_LDLIBS) $(LDLIBS)

tools/tokenize_prompt: tools/tokenize_prompt.c
	$(CC) $(TEST_CFLAGS) $(LLAMA_CFLAGS) -o $@ $< $(LLAMA_LDFLAGS) $(LLAMA_LDLIBS) $(LDLIBS)

tools/compare_logits: tools/compare_logits.c
	$(CC) $(TEST_CFLAGS) -o $@ $< -lm

test-tools-clean:
	rm -f $(TEST_TOOLS)

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

.PHONY: check-cache-large require-model clean check check-inference check-unit check-gpu check-context check-persistence check-prefill check-all check-snapshot-compat-host check-snapshot-compat \
	check-snapshot-compat-model check-kv check-kv-edges check-kv-load-bounds check-runtime check-serve golden test-tools test-tools-clean \
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

tests/test_vision_gpu: tests/test_vision_gpu.c xenolith.c xenolith.h vision_encoder.c vision_internal.h $(VISION_OBJS) format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ $< $(VISION_OBJS) format.o $(GPU_OBJ) $(LDLIBS)

check-vision: tests/test_vision_input tests/test_vision_gpu
	./tests/test_vision_input
	./tests/test_vision_gpu

tests/test_vision_session: tests/test_vision_session.c xenolith.c xenolith.h vision_encoder.c vision_internal.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ $< format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_text_batch_model: tests/test_text_batch_model.c xenolith.o xenolith.h format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ $< xenolith.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)

tests/test_weights_identity: tests/test_weights_identity.c xenolith.c xenolith.h format.h $(NUMERIC_SOURCES) $(VISION_OBJS) format.o $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ $< $(VISION_OBJS) format.o $(GPU_OBJ) $(LDLIBS)

# Exercise visual media replay with deterministic encoding and real storage.
tests/test_visual_replay: tests/test_visual_replay.c runtime.c runtime.h xenolith.h xenolith_internal.h $(NUMERIC_SOURCES) profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ)
	$(CC) $(CFLAGS) -I. -o $@ $< profile.o json.o conversation.o kvstore.o format.o $(VISION_OBJS) $(GPU_OBJ) $(LDLIBS)
