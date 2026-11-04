"""One declared CPU baseline for maintainer-built Linux engine artifacts."""

ENGINE_CFLAGS = "-O3 -march=x86-64-v3 -mavxvnni -std=c11 -Wall -Wextra -pthread"

# Linux /proc/cpuinfo spellings. x86-64-v1 is implied by the x86_64 platform
# check; include v2 and v3 requirements as well as the additional AVX-VNNI ISA.
# SSE3 is reported as pni, LAHF/SAHF as lahf_lm, and LZCNT as abm. Linux exposes
# usable AVX according to OS XSAVE support; there is no separate osxsave flag in
# /proc/cpuinfo that an installer should require.
ENGINE_CPU_FLAGS = (
    "cx16 lahf_lm popcnt pni ssse3 sse4_1 sse4_2 "
    "avx avx2 bmi1 bmi2 f16c fma abm movbe xsave avx_vnni"
)
