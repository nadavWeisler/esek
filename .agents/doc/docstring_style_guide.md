# Docstring Style Guide for ESEK

## Format: NumPy docstring style

All public functions, methods, and classes must follow NumPy docstring conventions.

## Required sections

### For functions and static methods:
1. **Summary line** — one sentence, imperative mood, no period at end if short
2. **Parameters** — every parameter with type annotation and valid range note
3. **Returns** — field-by-field for Result dataclasses
4. **Notes** — formula in LaTeX (`.. math::`) + academic reference
5. **References** — APA 7th edition
6. **Examples** — copy-pasteable Python with actual output

### For dataclasses (Results classes):
1. **Summary line**
2. **Attributes** — every field with type and description

### For base classes and abstract methods:
1. **Summary line**
2. **Notes** — design rationale if non-obvious

## LaTeX math in docstrings
Use `.. math::` directive:
```python
"""
Notes
-----
Cohen's d is computed as:

.. math:: d = \\frac{\\bar{x} - \\mu_0}{s}

where :math:`s` is the sample standard deviation.
"""
```

## Parameter type documentation
- Use Python types: `float`, `int`, `np.ndarray`, `list[float]`
- State valid ranges: `Must be in (0, 1)`, `Must be ≥ 2`
- For optional parameters: `optional` after the type, default in description

## Returns documentation for Result dataclasses
Document each field individually:
```python
Returns
-------
OneSampleTResults
    cohens_d : CohenD
        Cohen's d with `.value`, `.ci` (central), `.non_central_ci`, `.pivotal_ci`,
        `.standard_error`, and `.approximated_standard_error`.
    p_value : float
        Two-tailed p-value (capped at 0.99999).
```

## References format (APA 7)
```python
References
----------
Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.).
    Lawrence Erlbaum Associates.
Hedges, L. V. (1981). Distribution theory for Glass's estimator of effect size
    and related estimators. *Journal of Educational Statistics, 6*(2), 107–128.
    https://doi.org/10.2307/1164588
```

## Examples section
```python
Examples
--------
>>> from esek.calculator.one_sample_mean import OneSampleTTest
>>> result = OneSampleTTest.from_score(t_score=2.5, sample_size=30)
>>> print(f"Cohen's d = {result.cohens_d.value:.3f}")
Cohen's d = 0.463
>>> print(f"95% CI = [{result.cohens_d.ci.lower:.3f}, {result.cohens_d.ci.upper:.3f}]")
95% CI = [0.098, 0.821]
```

## What NOT to do
- Don't say "returns a dict" — specify the dataclass and each field
- Don't omit the formula — researchers need to trust the implementation
- Don't write "see source code" — the docstring IS the documentation
- Don't use `""" One liner """` for methods that have formulas — always expand
