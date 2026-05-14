# scaffold-agent: Example Request — Add One-Way ANOVA Effect Sizes

## Developer input
> "Add one-way ANOVA effect sizes to ESEK"

## What the scaffold-agent receives
```yaml
calculator_name: OneWayAnova
test_family: one_way_anova
effect_sizes:
  - EtaSquared
  - OmegaSquared
  - CohenF
formula_description: >
  From an F-statistic and design parameters (k groups, N total):
  - eta^2 = SS_between / SS_total = (df_between * F) / (df_between * F + df_within)
  - omega^2 = (df_between * (F - 1)) / (df_between * (F - 1) + N)
  - Cohen's f = sqrt(eta^2 / (1 - eta^2))
reference: "Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences (2nd ed.)."
known_values:
  - inputs: {F: 4.5, k: 3, N: 60}
    outputs: {eta_squared: 0.1346, omega_squared: 0.1111, cohens_f: 0.3947}
```

## What the scaffold-agent produces

### File 1: `src/esek/utils/effect_sizes.py` (additions)
```python
class EtaSquared(EffectSize):
    """η² effect size for ANOVA."""
    def __init__(self, value, ci_lower, ci_upper, standard_error):
        super().__init__(value, ci_lower, ci_upper, standard_error)
        self.effect_size_name = "Eta Squared"
        self.update_statistical_line()

class OmegaSquared(EffectSize):
    """ω² (bias-corrected) effect size for ANOVA."""
    ...

class CohenF(EffectSize):
    """Cohen's f effect size for ANOVA."""
    ...
```

### File 2: `src/esek/calculator/one_way_anova/__init__.py`
```python
from .one_way_anova import OneWayAnovaTest, OneWayAnovaResults
__all__ = ["OneWayAnovaTest", "OneWayAnovaResults"]
```

### File 3: `src/esek/calculator/one_way_anova/one_way_anova.py`
Full implementation with from_score, from_parameters, from_data stub.

### File 4: `tests/test_one_way_anova.py`
Smoke tests + numerical fixture stubs ready for numerical-validation-agent.

### PR opened
Title: `feat(calculator): add OneWayAnova effect sizes (eta², ω², Cohen's f)`
Draft: true
Labels: calculator, tests
