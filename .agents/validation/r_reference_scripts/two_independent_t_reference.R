# R Reference Script: Two-Independent-Samples T-Test Validation
# Used by numerical-validation-agent for
# src/esek/calculator/two_independent_mean/two_independent_t.py
#
# Run with: Rscript two_independent_t_reference.R
# Packages: effectsize, MBESS
#
# Install: install.packages(c("effectsize", "MBESS"))

library(effectsize)
library(MBESS)

cat("=== Two-Independent T-Test Reference Values for ESEK ===\n\n")

# Test case 1: equal N, equal variance
t <- 2.5; n1 <- 30; n2 <- 30; conf <- 0.95
df <- n1 + n2 - 2
d <- t * sqrt(1/n1 + 1/n2)
cat(sprintf("Case 1: t=%.1f, n1=%d, n2=%d\n", t, n1, n2))
cat(sprintf("  cohens_d  = %.6f\n", d))
res <- MBESS::ci.smd(smd=d, n.1=n1, n.2=n2, conf.level=conf)
cat(sprintf("  ci_lower  = %.6f\n", res$Lower.Conf.Limit.smd))
cat(sprintf("  ci_upper  = %.6f\n", res$Upper.Conf.Limit.smd))
p <- 2 * pt(-abs(t), df=df)
cat(sprintf("  p_value   = %.6f\n", p))
cat(sprintf("  df        = %d\n\n", df))

# Test case 2: unequal N
t <- 1.8; n1 <- 20; n2 <- 40; conf <- 0.95
df <- n1 + n2 - 2
d <- t * sqrt(1/n1 + 1/n2)
cat(sprintf("Case 2: t=%.1f, n1=%d, n2=%d\n", t, n1, n2))
cat(sprintf("  cohens_d  = %.6f\n", d))
res <- MBESS::ci.smd(smd=d, n.1=n1, n.2=n2, conf.level=conf)
cat(sprintf("  ci_lower  = %.6f\n", res$Lower.Conf.Limit.smd))
cat(sprintf("  ci_upper  = %.6f\n", res$Upper.Conf.Limit.smd))
p <- 2 * pt(-abs(t), df=df)
cat(sprintf("  p_value   = %.6f\n\n", p))

# Test case 3: large effect
t <- 4.0; n1 <- 15; n2 <- 15; conf <- 0.95
df <- n1 + n2 - 2
d <- t * sqrt(1/n1 + 1/n2)
cat(sprintf("Case 3: t=%.1f, n1=%d, n2=%d\n", t, n1, n2))
cat(sprintf("  cohens_d  = %.6f\n", d))
res <- MBESS::ci.smd(smd=d, n.1=n1, n.2=n2, conf.level=conf)
cat(sprintf("  ci_lower  = %.6f\n", res$Lower.Conf.Limit.smd))
cat(sprintf("  ci_upper  = %.6f\n", res$Upper.Conf.Limit.smd))
p <- 2 * pt(-abs(t), df=df)
cat(sprintf("  p_value   = %.6f\n\n", p))
