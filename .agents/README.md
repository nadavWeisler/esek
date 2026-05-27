# ESEK Custom AI Agents

This directory contains the configuration and prompts for all specialized AI agents designed for the **ESEK (Effect Size Estimation Kit)** project.

## Quick Start

Each agent lives in its own subdirectory. To invoke an agent, supply:
1. Its `agent.yml` config to your AI runtime (GitHub Copilot, OpenAI Assistants, LangChain, etc.)
2. The `system_prompt.txt` as the system message
3. Any `context_files` listed in `agent.yml` as additional context

---

## Agent Directory

| Agent | Trigger | Model | Primary Job |
|---|---|---|---|
| [scaffold](#scaffold-agent) | On-demand | Cloud | Generate new calculator boilerplate |
| [validation](#numerical-validation-agent) | CI / On-demand | Cloud | Verify numerical correctness |
| [migration](#migration-agent) | On-demand | Local | Complete `Calculator/` → `calculator/` migration |
| [doc](#doc-agent) | Continuous / On-demand | Local | Docstrings, docs site, changelog |
| [regression-guard](#regression-guard-agent) | CI | Local | Golden-file numerical regression detection |
| [roadmap](#roadmap-agent) | On-demand | Cloud | Idea → GitHub Milestone + Issues |
| [release](#release-agent) | On-demand | Local | Prepare and gate releases to PyPI |
| [interpretation](#interpretation-agent) | On-demand | Cloud | APA-7 statistical report strings |

---

## Agent Descriptions

### scaffold-agent
Eliminates boilerplate. Every new effect-size calculator follows the same rigid pattern (`@dataclass Results` + `class Test(AbstractTest)` with `from_score / from_parameters / from_data`). Give the agent a formula and it generates the full calculator file, test file, `__init__.py` updates, and migration shim.

### numerical-validation-agent
Upgrades tests from smoke tests (isinstance checks) to ground-truth numerical fixtures verified against R packages (`effectsize`, `MBESS`, `effsize`). Runs automatically in CI on any change to `calculator/` or `utils/`.

### migration-agent
Completes the structural migration from `src/esek/Calculator/` (legacy uppercase) to `src/esek/calculator/` (canonical lowercase). The `Proportions/` submodule and a few other files still need migrating. Leaves `DeprecationWarning` shims in the old location to preserve backward compatibility.

### doc-agent
Generates NumPy-style docstrings, a MkDocs documentation site, and a `CHANGELOG.md`. For a scientific library, documentation IS the product — researchers need exact formulas, assumptions, and working code examples.

### regression-guard-agent
Serializes representative calculator outputs to `tests/golden/*.json` and re-checks them on every CI run. Ensures that NumPy/SciPy upgrades or internal refactors never silently change the numbers researchers cite in papers.

### roadmap-agent
Takes a one-sentence feature idea and decomposes it into a GitHub Milestone with labelled, prioritized, dependency-linked Issues — one per file/concern. Designed for a solo developer who wants to describe a goal and immediately have a structured backlog.

### release-agent
Runs the release checklist: all tests green, golden files unchanged, changelog updated, version semantics correct, GitHub Release draft created. Requires one manual approval before pushing the tag and triggering the PyPI publish workflow.

### interpretation-agent
Implements the unused `utils/texts.py` interpretive engine. Produces APA 7th edition statistical report sentences, plain-English summaries, and LaTeX-formatted statistical lines from any `Results` dataclass instance.

---

## Recommended Workflow for Solo Developer

```
You:          "Add ANOVA effect sizes"
                       │
              roadmap-agent
                       │ creates Milestone + Issues
                       ▼
              scaffold-agent          (per implementation issue)
                       │ opens draft PR
                       ▼
      numerical-validation-agent      (runs in CI on the PR)
      regression-guard-agent          (runs in CI on the PR)
              doc-agent               (runs continuously, adds docstrings)
                       │
              You: review + approve
                       │
              release-agent           (when milestone is complete)
```

---

## Global Configuration

See `config.yml` for model endpoints and shared settings.
