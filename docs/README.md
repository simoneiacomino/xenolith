# Xenolith documentation

- [Contributing](../CONTRIBUTING.md): pull requests, commit messages and validation.
- [Installation](../installer/README.md): guided setup, existing models and clients.
- [Engine releases and distributions](../installer/RELEASING.md): versions,
  package qualification and publication.
- [Build from source](build.md): dependencies, model setup and installation.
- [CLI](cli.md): commands, options, environment variables and service paths.
- [C API](api.md): engine configuration, sessions and resource lifetimes.
- [Wire protocol](protocol.md): requests, events and client integration.
- [Vision](vision.md): image usage, identities and persistence formats.
- [Performance tuning](tuning.md): automatic CPU profiles, manual maps and hardware constraints.
- [Testing](testing.md): host, GPU, model and vision checks.
- [Benchmarks](../bench/README.md): methods and selected results.

Commands in these pages run from the repository root unless stated otherwise.

The [numerical fidelity workflow](testing.md#numerical-fidelity) uses optional
Python environments and reusable Google FP64 archives. See the
[legacy check migration](quality-migration.md) for API and tooling changes.
- [Experimental activation steering](../STEERING.md): local vector extraction, inference controls, persistence, and evaluation.
