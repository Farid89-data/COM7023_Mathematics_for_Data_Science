# Hypothesis Testing

## Overview

Hypothesis testing is a formal statistical procedure for making decisions based on data. It provides a framework for determining whether observed results are statistically significant or could have occurred by chance.

## Mathematical Concepts

### The Hypothesis Testing Framework

1. **State Hypotheses**
   - $H_0$: Null hypothesis (status quo)
   - $H_1$ or $H_a$: Alternative hypothesis

2. **Choose Significance Level**
   - $\alpha$ = probability of Type I error (typically 0.05)

3. **Calculate Test Statistic**
   - Depends on the type of test

4. **Make Decision**
   - Compare p-value to $\alpha$, or
   - Compare test statistic to critical value

### Common Test Statistics

**Z-test (known variance):**
$$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

**t-test (unknown variance):**
$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}$$

**Chi-square test:**
$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

### Confidence Intervals

A $(1-\alpha)$ confidence interval:
$$\bar{x} \pm z_{\alpha/2} \times \frac{\sigma}{\sqrt{n}}$$

## Applications in Data Science

1. **A/B Testing**: Compare treatment effects
2. **Feature Selection**: Test predictor significance
3. **Model Validation**: Test model assumptions
4. **Quality Control**: Monitor process parameters

## Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `hypothesis_testing_marking_matrix.py` | Marking Matrix | Various statistical tests |
| `hypothesis_testing_triplet_births.py` | Multiple Births | Time series hypothesis tests |

## References

Arden University (2026) *COM7023 Mathematics for Data Science Marking Matrix*. Arden University.

Human Multiple Births Database (2024) *FRA_InputData_25.11.2024.xlsx*. Available at: https://www.twinbirths.org/en/data-metadata/ (Accessed: 25 November 2024).

Wasserman, L. (2004) *All of Statistics*. Springer.
