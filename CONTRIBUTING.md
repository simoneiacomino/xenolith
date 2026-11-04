# Contributing to Xenolith

## Pull requests

Target `main` from a short-lived branch. Discuss substantial design or
compatibility changes before implementation. Keep each PR focused and describe
the problem, resulting behavior, checks performed and any migration needs. Link
dependent changes in other repositories and state their integration order.

Keep `main` buildable. Before merging, require the relevant checks to pass and
resolve open review discussions; additional review is optional. Use pull requests
for ordinary changes and avoid force-pushing shared `main`. Document any urgent
exception and its validation. Use merge commits by default; agree any feature
squash with its contributors and preserve authorship and attribution trailers.

## Commit messages

Use Conventional Commits for new commits and PR titles: a lowercase type,
optional scope and concise English imperative description, for example:

```text
fix(config): Preserve explicit command options
```

Use `feat`, `fix`, `perf`, `refactor`, `docs`, `test`, `build`, `ci` or `chore`.
Explain nontrivial motivation in the body. Mark incompatible changes with `!`
and document migration, including during `0.x`. Keep each commit coherent.
Imported history and generated merge/revert messages retain their messages.

## Validation

Choose checks for the affected behavior; see [testing](docs/testing.md).
CLI/configuration changes need configuration and CLI checks. Installer or package
changes need fixtures and, for layout or activation changes, actual package
installation checks. Protocol changes need coverage of affected client contracts.
Inference changes need numerical and representative model checks. Performance
claims need matched controls, source/artifact identities and reproducible results.
Documentation changes need link or build checks where applicable.

Validate against current `main` and recheck after material base changes. Record
skipped or unavailable checks explicitly. Simulated hardware and client tests
do not establish real-model or GPU compatibility.

## Documentation and releases

Keep user guides, contributor instructions and reproducible benchmarks public.
Keep internal execution diaries, raw validation reports, campaign planning,
credentials and private operational details outside the repository and packages.
Preserve existing license notices and contributor attribution.

Merging into `main` does not publish a release. See the
[release guide](installer/RELEASING.md) for engine versions, compatibility,
artifact qualification and distribution publication.
