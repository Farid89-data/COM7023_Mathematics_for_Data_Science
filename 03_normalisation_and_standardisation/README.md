# Normalisation and Standardisation

## Overview

Data normalisation and standardisation are essential preprocessing techniques that transform data to a common scale, enabling fair comparison and improving machine learning algorithm performance.

## Mathematical Concepts

### Min-Max Normalisation

Scales data to a fixed range, typically [0, 1]:

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Z-Score Standardisation

Transforms data to have mean 0 and standard deviation 1:

$$z = \frac{x - \mu}{\sigma}$$

### Decimal Scaling

Scales by powers of 10:

$$x_{scaled} = \frac{x}{10^j}$$

where j is the smallest integer such that max(|x_scaled|) < 1.

## Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `normalisation_marking_matrix.py` | Marking Matrix | Normalise derived numerical features |
| `normalisation_triplet_births.py` | Multiple Births | Normalise demographic time series |

## References

Arden University (2026) *COM7023 Mathematics for Data Science Marking Matrix*. Arden University.

Human Multiple Births Database (2024) *FRA_InputData_25.11.2024.xlsx*. Available at: https://www.twinbirths.org/en/data-metadata/ (Accessed: 25 November 2024).
