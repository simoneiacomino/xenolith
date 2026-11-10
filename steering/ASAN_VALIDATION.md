# ASan host validation

The previous OneAgent/AddressSanitizer initialization blocker is resolved for
isolated test execution on `xenolith-ptl`. No host preload configuration is changed.

```sh
tools/without_system_preload.sh make -j4 check-host
```

Result: exit status 0. The ASan bounds harness passed all 152 valid cases and
rejected all three deliberately broken variants as out-of-bounds reads. The
remaining host checks also completed successfully. Full output is recorded in
`results/checks/xenolith-asan-isolated-host.log`.

The tested kernel source SHA-256 is
`e8d19d9b916d1912eea40facfc7f6ef99f49173faa1b626b508a8b2f6549ad07`.
The source matches the kernel in the current steering branch. The native checkout
includes activation steering and both reasoning/response prefix modes.

The wrapper uses bubblewrap to mask `/etc/ld.so.preload` inside a private mount
namespace. It does not disable AddressSanitizer or weaken the mutation checks.
Normal test execution outside the wrapper is still subject to the host preload.

This result establishes `check-host`, not full `check-all`. The previously
reported full-model CPU/GPU prefill discrepancy is a separate unresolved issue.
