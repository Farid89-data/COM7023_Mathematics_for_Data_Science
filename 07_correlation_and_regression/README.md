# Correlation and Regression

## Overview

Correlation and regression analysis are fundamental statistical techniques for understanding relationships between variables. Correlation measures the strength and direction of association, while regression models the functional relationship for prediction.

## Mathematical Concepts

### Correlation Coefficient

The Pearson correlation coefficient measures linear association:

$$r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

Properties of r:

- $-1 \leq r \leq 1$
- $r = 1$: Perfect positive correlation
- $r = -1$: Perfect negative correlation
- $r = 0$: No linear correlation

### Simple Linear Regression

The linear regression model:

$$y = \beta_0 + \beta_1 x + \varepsilon$$

Least squares estimates:

$$\hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i - \bar{x})^2}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

### Coefficient of Determination

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

## Applications in Data Science

1. **Feature Selection**: Identify important predictors
2. **Trend Analysis**: Model temporal patterns
3. **Prediction**: Forecast future values
4. **Causal Inference**: Understand variable relationships

## Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `correlation_regression_marking_matrix.py` | Marking Matrix | Correlation analysis |
| `correlation_regression_triplet_births.py` | Multiple Births | Regression modelling |

## References

Arden University (2026) *COM7023 Mathematics for Data Science Marking Matrix*. Arden University.

Human Multiple Births Database (2024) *FRA_InputData_25.11.2024.xlsx*. Available at: https://www.twinbirths.org/en/data-metadata/ (Accessed: 25 November 2024).

Freedman, D.A. (2009) *Statistical Models: Theory and Practice*. Cambridge University Press.
