# ESEK — Effect Size Estimation Kit

**ESEK** is a Python library for computing and converting effect sizes across common statistical designs (t-tests, paired tests, non-parametric tests, proportions, and more).

> ⚠️ **Work in progress**  
> The API may change as statistical methods are being integrated.

---

## Installation

```bash
pip install esek
```

Or install from source:

```bash
git clone https://github.com/nadavWeisler/esek.git
cd esek
pip install -e .
```

**Requirements:** Python ≥ 3.10, NumPy ≥ 2.0, SciPy ≥ 1.13, statsmodels ≥ 0.14

---

## Quickstart

### Effect Size Calculation

```python
# Two independent groups (t-test)
from esek.calculators.two_independent_mean.two_independent_t import TwoIndependentTTests

result = TwoIndependentTTests.from_parameters(
    sample_mean_1=5.2,
    sample_mean_2=4.0,
    sample_sd_1=1.1,
    sample_sd_2=1.0,
    sample_size_1=30,
    sample_size_2=30,
    population_mean_diff=0,
    confidence_level=0.95,
)

print(result.cohens_d.value)           # Cohen's d
print(result.cohens_d.ci.lower)        # CI lower bound
print(result.cohens_d.ci.upper)        # CI upper bound
print(result.hedges_g.value)           # Hedges' g (bias-corrected)
```

### Effect Size Conversion

```python
from esek import EffectSizeConverter

# Convert Cohen's d to Pearson r
result = EffectSizeConverter.d_to_r(d=0.5, n1=30, n2=30)
print(result.output_value)   # → 0.243

# Convert d to odds ratio
or_result = EffectSizeConverter.d_to_odds_ratio(d=0.5)
print(or_result.output_value)  # → 2.477

# Fisher z transformation
z_result = EffectSizeConverter.r_to_fisher_z(r=0.6)
r_back = EffectSizeConverter.fisher_z_to_r(z_result.output_value)
```

### Confidence Intervals

```python
from esek.confidence_intervals import central_ci_one_sample, fisher_z_ci

# CI for Cohen's d (one-sample)
ci_low, ci_high, se = central_ci_one_sample(effect_size=0.5, sample_size=30, confidence_level=0.95)

# Fisher z CI for correlation
ci_low, ci_high = fisher_z_ci(r=0.6, n=50, confidence_level=0.95)
```

---

## Supported Effect Sizes

| Design | Effect Sizes |
|--------|-------------|
| One-sample t / z | Cohen's d, Hedges' g, CLES |
| Two independent groups | Cohen's d, Hedges' g, Glass's Δ, Ratio of Means, Cliff's delta, VDA, U1/U3 |
| Two paired groups | Cohen's dav, gav, drm, grm, rank-biserial, robust measures |
| Proportions | Cohen's h, g, Phi, OR, RR, Cramer's V |
| Converters | d↔r, d↔OR, r↔Fisher z, OR↔d |

---

## Interpretation Warning

Effect sizes are statistical summaries. They do not determine whether a finding is practically important — this judgment requires domain knowledge, study design context, and replication. Interpret them in context.

---

## Testing

```bash
pytest
python -m compileall src/
```

---

## Package Structure

```
src/esek/
    core/          ← exceptions, validation, type aliases
    results/       ← frozen dataclass result objects
    calculators/   ← statistical calculators (t, z, aparametric, proportions)
    converters/    ← effect size conversion functions
    confidence_intervals/  ← CI methods
    utils/         ← math helpers, distribution helpers
```

---

## License

GPL-3.0

