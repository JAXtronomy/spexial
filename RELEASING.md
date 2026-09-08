# Releasing `spexial`

`spexial` is a single package, released from tags. There is no `CHANGELOG.md`; GitHub Releases are the changelog.

## How versioning works

The version is derived from git tags by [`hatch-vcs`](https://github.com/ophiuscience/hatch-vcs), configured in `pyproject.toml`:

- Tags must match `v*` (`describe_command` in `[tool.hatch.version.raw-options.scm.git]`).
- `local_scheme = "no-local-version"`, so builds off a tag produce a clean, PyPI-uploadable version.
- The version is written to `src/spexial/_version.py` at build time. That file is generated and gitignored; `_version.pyi` is the checked-in stub that type checkers see.

Between releases you will see versions like `0.2.dev14`. That is expected.

## Cutting a release

1. **Check `main` is green.** The `CI Pass` job must be passing on the commit you intend to tag.

2. **Decide the version.** Semantic versioning. For a numerics library, note that a change to a function's *numerical output* beyond its documented tolerance is a breaking change even if the signature is unchanged — treat it as such.

3. **Tag and push.**

   ```bash
   git switch main && git pull
   git tag -a v0.2.0 -m "v0.2.0"
   git push origin v0.2.0
   ```

4. **CD runs automatically.** The tag push triggers `.github/workflows/cd.yml`, which:
   - builds the sdist and wheel with `hynek/build-and-inspect-python-package`,
   - attaches a build-provenance attestation,
   - publishes to PyPI via trusted publishing (environment `pypi`, no API token).

5. **Write the release notes.** GitHub drafts them from the merged PRs using the categorisation in `.github/release.yml`. Edit for readability, and call out explicitly:
   - any change to numerical output or accepted domain,
   - any newly supported or newly documented-as-unsupported regime.

## First release

No tags exist yet, so `hatch-vcs` currently reports a `0.1.dev*` placeholder. The first `v0.1.0` tag establishes the baseline.

Before it, confirm:

- [ ] The PyPI trusted publisher for `spexial` is configured, with environment name `pypi`.
- [ ] `CODECOV_TOKEN` and `CODSPEED_TOKEN` repository secrets are set.
- [ ] GitHub Pages is enabled with source = GitHub Actions (done).
- [ ] The author list in `CITATION.cff` is confirmed — it currently carries a TODO about the LINX contributor.

## If a release goes wrong

PyPI does not allow re-uploading a version. Yank the bad release on PyPI, fix forward, and tag a new patch version. Do not delete and re-push a tag that CD has already acted on.
