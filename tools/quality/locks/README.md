These are resolved, hashed installation profiles for Linux x86_64 and Python 3.12.
They pin transitive dependencies as well as the direct versions in pyproject.toml.
They include the build backend to avoid an implicit build dependency resolution.
Only wheel artifacts are installed by setup_environment.py. No environment is
selected or installed automatically by tests or by the C build.

Regenerate deliberately with uv pip compile --generate-hashes --no-header,
using pyproject.toml and locks/build.in, and --extra reference or --extra
qualification or --extra corpus for the respective profiles. Record the resolver
version, platform, installation and qualification evidence for any update. The
corpus profile pins optional prompt-generation dependencies separately; none are
required to read frozen corpora or compare numerical results.

The reference and qualification environments are separate. Installing a profile
is a software smoke check, not numerical qualification of that runtime. Archives retain the producer and environment identities used to generate them.

The qualification lock selects torch 2.14.0+cpu from the official PyTorch CPU
index using qualification-cpu.in. Resolve it with --extra-index-url
https://download.pytorch.org/whl/cpu and --index-strategy unsafe-best-match, then
retain the explicit index line in the pip lock. Hash checking pins accepted
artifacts from either index. Generic PyPI Torch installs CUDA dependencies and
is deliberately not the canonical qualification installation.
