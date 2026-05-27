# Release Checklist

Complete every item before pushing the release tag.
The release-agent will check each item and block release until all are green.

## Pre-release checks

- [ ] **CI green**: All `pytest` runs passing on Python 3.10, 3.11, 3.12
- [ ] **Lint clean**: `flake8 . --select=E9,F63,F7,F82` reports zero errors
- [ ] **No regressions**: Zero open GitHub issues with label `regression`
- [ ] **Golden files current**: `tests/golden/` files match current `main` outputs
- [ ] **CHANGELOG updated**: `CHANGELOG.md` has an entry for the new version
- [ ] **Version consistent**: `pyproject.toml` `[project]` version matches the tag (if static)
- [ ] **Milestone clean**: All milestone issues are closed or explicitly deferred

## Release preparation

- [ ] **GitHub Release draft** created with auto-generated notes
- [ ] **Breaking changes** documented in release notes (for minor/major bumps)
- [ ] **Upgrade instructions** written (for major bumps or API changes)
- [ ] **README** reviewed for accuracy (remove WIP warning at v1.0.0)

## Post-approval (developer must approve before these run)

- [ ] Git tag pushed: `v{version}` → triggers `python-publish.yml`
- [ ] GitHub Release published (draft → published)
- [ ] Milestone closed

## Version decision guide

| Change type | Version bump |
|---|---|
| Bug fix, typo, docs only | patch |
| New `from_data()` on existing calculator | patch |
| New effect-size calculator or CI method | minor |
| New statistical test family (e.g. ANOVA) | minor |
| Changed method signature (breaking) | major |
| Removed public class or method | major |
| First stable release | major (→ 1.0.0) |
