# APA 7th Edition Statistical Reporting Examples
# Reference for interpretation-agent

## One-sample t-test
> A one-sample t-test revealed a statistically significant medium effect,
> *t*(29) = 2.50, *p* = .013, *d* = 0.46, 95% CI [0.10, 0.82].

> A one-sample t-test did not reveal a statistically significant effect,
> *t*(49) = 1.23, *p* = .224, *d* = 0.17, 95% CI [−0.11, 0.46].

## Two independent samples t-test
> An independent-samples t-test revealed a statistically significant large effect,
> *t*(58) = 4.12, *p* < .001, *d* = 1.06, 95% CI [0.54, 1.58].

## Two paired samples t-test
> A paired-samples t-test revealed a statistically significant small effect,
> *t*(24) = 2.08, *p* = .048, *d*_av = 0.42, 95% CI [0.003, 0.831].

## One-sample z-test
> A one-sample z-test revealed a statistically significant medium effect,
> *z* = 2.10, *p* = .036, *d* = 0.42, 95% CI [0.03, 0.81].

## One-sample proportion (Cohen's h)
> A one-sample proportion test revealed a statistically significant medium effect,
> *z* = 2.34, *p* = .019, *h* = 0.47, 95% CI [0.08, 0.86].

## ANOVA (when implemented)
> A one-way ANOVA revealed a statistically significant medium effect,
> *F*(2, 57) = 4.50, *p* = .015, η² = 0.14, 95% CI [0.02, 0.27].

---

## Formatting rules

| Element | Format | Example |
|---|---|---|
| p-value ≥ .001 | `p = .XYZ` (no leading zero, 3 decimal places) | `p = .013` |
| p-value < .001 | `p < .001` | `p < .001` |
| Effect size | 2 decimal places | `d = 0.46` |
| CI bounds | 2 decimal places in brackets | `[0.10, 0.82]` |
| t-statistic | `t(df) = X.XX` | `t(29) = 2.50` |
| z-statistic | `z = X.XX` (no df) | `z = 1.96` |
| F-statistic | `F(df1, df2) = X.XX` | `F(2, 57) = 4.50` |
| Negative values | Include minus sign | `d = −0.34` (use Unicode minus U+2212) |

## Magnitude labels (Cohen, 1988)

### Cohen's d / Hedges' g
| |d|| Label |
|---|---|
| < 0.20 | trivial |
| 0.20–0.49 | small |
| 0.50–0.79 | medium |
| ≥ 0.80 | large |

### Cohen's f (ANOVA)
| f | Label |
|---|---|
| < 0.10 | trivial |
| 0.10–0.24 | small |
| 0.25–0.39 | medium |
| ≥ 0.40 | large |

### Cohen's h (proportions)
| h | Label |
|---|---|
| < 0.20 | trivial |
| 0.20–0.49 | small |
| 0.50–0.79 | medium |
| ≥ 0.80 | large |
