# Descriptive Statistics

## Overview

This section demonstrates the calculation and interpretation of descriptive statistics, which form the foundation of data analysis in data science.

## Mathematical Concepts

### Measures of Central Tendency

**Arithmetic Mean:**
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

**Median:** The middle value when data is ordered.

**Mode:** The most frequently occurring value.

### Measures of Dispersion

**Variance (Sample):**
$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

**Standard Deviation:**
$$s = \sqrt{s^2}$$

**Range:**
$$\text{Range} = x_{max} - x_{min}$$

**Interquartile Range:**
$$\text{IQR} = Q_3 - Q_1$$

## Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `descriptive_statistics_marking_matrix.py` | Marking Matrix | Text-based statistical analysis |
| `descriptive_statistics_triplet_births.py` | Multiple Births | Numerical descriptive statistics |

## References

Arden University (2026) *COM7023 Mathematics for Data Science Marking Matrix*. Arden University.

Human Multiple Births Database (2024) *FRA_InputData_25.11.2024.xlsx*. Available at: https://www.twinbirths.org/en/data-metadata/ (Accessed: 25 November 2024).
