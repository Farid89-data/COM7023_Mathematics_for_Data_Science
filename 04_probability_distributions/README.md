# Probability Distributions

## Overview

This section covers probability theory and statistical distributions fundamental to data science. Understanding probability distributions enables prediction, inference, and decision-making under uncertainty.

## Mathematical Concepts

### Discrete Distributions

**Poisson Distribution:**
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Used for modelling rare events when:
- Events occur independently
- Average rate λ is constant
- Events occur one at a time

### Continuous Distributions

**Normal (Gaussian) Distribution:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

### Maximum Likelihood Estimation

For Poisson distribution:
$$\hat{\lambda}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i$$

## Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `probability_distributions_marking_matrix.py` | Marking Matrix | Categorical probability analysis |
| `poisson_distribution_triplet_births.py` | Multiple Births | Poisson modelling of rare events |

## References

Arden University (2026) *COM7023 Mathematics for Data Science Marking Matrix*. Arden University.

Human Multiple Births Database (2024) *FRA_InputData_25.11.2024.xlsx*. Available at: https://www.twinbirths.org/en/data-metadata/ (Accessed: 25 November 2024).
