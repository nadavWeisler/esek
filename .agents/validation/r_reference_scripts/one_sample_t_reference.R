# R Reference Script: One-Sample T-Test Validation
# Used by numerical-validation-agent to generate ground-truth values for
# src/esek/calculator/one_sample_mean/one_sample_t.py
#
# Run with: Rscript one_sample_t_reference.R
# Packages: effectsize, MBESS
#
# Install: install.packages(c("effectsize", "MBESS"))

library(effectsize)
library(MBESS)

cat("=== One-Sample T-Test Reference Values for ESEK ===\n\n")

# Test case 1: typical
t <- 2.5; n <- 30; conf <- 0.95
d <- t / sqrt(n - 1)
cat(sprintf("Case 1: t=%.1f, n=%d\n", t, n))
cat(sprintf("  cohens_d  = %.6f\n", d))
res <- MBESS::ci.smd(smd=d, n.1=n, conf.level=conf)
cat(sprintf("  ci_lower  = %.6f\n", res$Lower.Conf.Limit.smd))
cat(sprintf("  ci_upper  = %.6f\n", res$Upper.Conf.Limit.smd))
p <- 2 * pt(-abs(t), df=n-1)
cat(sprintf("  p_value   = %.6f\n", p))
cat(sprintf("  df        = %d\n\n", n-1))

# Test case 2: small effect
t <- 0.8; n <- 50; conf <- 0.95
d <- t / sqrt(n - 1)
cat(sprintf("Case 2: t=%.1f, n=%d\n", t, n))
cat(sprintf("  cohens_d  = %.6f\n", d))
res <- MBESS::ci.smd(smd=d, n.1=n, conf.level=conf)
cat(sprintf("  ci_lower  = %.6f\n", res$Lower.Conf.Limit.smd))
cat(sprintf("  ci_upper  = %.6f\n", res$Upper.Conf.Limit.smd))
p <- 2 * pt(-abs(t), df=n-1)
cat(sprintf("  p_value   = %.6f\n\n", p))

# Test case 3: large effect, small N
t <- 3.8; n <- 10; conf <- 0.95
d <- t / sqrt(n - 1)
cat(sprintf("Case 3: t=%.1f, n=%d\n", t, n))
cat(sprintf("  cohens_d  = %.6f\n", d))
res <- MBESS::ci.smd(smd=d, n.1=n, conf.level=conf)
cat(sprintf("  ci_lower  = %.6f\n", res$Lower.Conf.Limit.smd))
cat(sprintf("  ci_upper  = %.6f\n", res$Upper.Conf.Limit.smd))
p <- 2 * pt(-abs(t), df=n-1)
cat(sprintf("  p_value   = %.6f\n\n", p))

# Test case 4: negative t-score
t <- -1.96; n <- 100; conf <- 0.95
d <- t / sqrt(n - 1)
cat(sprintf("Case 4: t=%.2f, n=%d\n", t, n))
cat(sprintf("  cohens_d  = %.6f\n", d))
res <- MBESS::ci.smd(smd=d, n.1=n, conf.level=conf)
cat(sprintf("  ci_lower  = %.6f\n", res$Lower.Conf.Limit.smd))
cat(sprintf("  ci_upper  = %.6f\n", res$Upper.Conf.Limit.smd))
p <- 2 * pt(-abs(t), df=n-1)
cat(sprintf("  p_value   = %.6f\n\n", p))

# Test case 5: 99% confidence level
t <- 2.5; n <- 30; conf <- 0.99
d <- t / sqrt(n - 1)
cat(sprintf("Case 5: t=%.1f, n=%d, conf=%.2f\n", t, n, conf))
cat(sprintf("  cohens_d  = %.6f\n", d))
res <- MBESS::ci.smd(smd=d, n.1=n, conf.level=conf)
cat(sprintf("  ci_lower  = %.6f\n", res$Lower.Conf.Limit.smd))
cat(sprintf("  ci_upper  = %.6f\n\n", res$Upper.Conf.Limit.smd))
