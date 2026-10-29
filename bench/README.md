# Benchmarks

- [Prefill and decode comparison](compare_pp_tg.md): historical laptop comparison with llama.cpp, including data and reproduction instructions.
- [CPU worker measurements](cpu-workers.md): worker scaling on two machines and a separate integration regression check, with selected machine-readable measurements.

Results describe the recorded builds and conditions. Different protocols and dates should not be combined into a single ranking. Build and test new configurations on the target machine; see [engine configuration](../docs/configuration.md).

For a new benchmark contribution, include the engine commit and local changes; CPU, GPU device ID, memory, OS, driver/runtime and compiler; model identity and hash; exact command, worker map and context; inputs and timing boundaries; warmup, cache and power conditions; all samples, failures and correctness checks. State how results were summarized and keep distinct protocols separate. Review logs for private paths or machine identifiers before sharing. Submit a concise report and selected machine-readable results in `bench/`, with large raw evidence hosted separately when appropriate.
