---
name: Test issue
about: Add numerical fixture tests for a calculator
labels: tests
---

## Goal
<!-- One sentence: what tests need to be added -->

## Acceptance Criteria
- [ ] At least 5 `pytest.mark.parametrize` test cases with R-verified expected values
- [ ] Each fixture has: inputs, expected outputs, tolerance, and R source reference comment
- [ ] `rtol=1e-4` (or tighter for simple closed-form formulas)
- [ ] Tests pass in CI
- [ ] No `NaN` or `inf` outputs accepted silently

## Calculator being tested
<!-- Path to calculator file -->

## R reference
<!-- R package + function to use for verification -->

## Blocked by
<!-- #issue_number (calculator must exist first) -->
