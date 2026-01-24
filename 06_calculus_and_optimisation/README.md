# Calculus and Optimisation

## Overview

Calculus provides the mathematical tools for understanding change and accumulation, while optimisation techniques find the best solutions to problems. These concepts are fundamental to machine learning, particularly in training models through gradient descent.

## Mathematical Concepts

### Derivatives

The derivative measures the instantaneous rate of change of a function:

$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Common Derivative Rules:**

- Power Rule: $\frac{d}{dx}x^n = nx^{n-1}$

- Chain Rule: $\frac{d}{dx}f(g(x)) = f'(g(x)) \cdot g'(x)$

- Product Rule: $\frac{d}{dx}[f(x)g(x)] = f'(x)g(x) + f(x)g'(x)$

### Integrals

The integral represents accumulation or area under a curve:

$$\int_a^b f(x)\,dx = F(b) - F(a)$$

where $F'(x) = f(x)$ (Fundamental Theorem of Calculus).

### Gradient Descent

An iterative optimisation algorithm to find local minima:

$$\theta_{t+1} = \theta_t - \alpha \nabla J(\theta_t)$$

where $\alpha$ is the learning rate and $\nabla J$ is the gradient of the cost function.

## Applications in Data Science

1. **Model Training**: Minimise loss functions using gradient descent
2. **Feature Engineering**: Rate of change analysis
3. **Probability**: Integration for continuous distributions
4. **Neural Networks**: Backpropagation uses chain rule

## Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `calculus_optimisation_marking_matrix.py` | Marking Matrix | Derivatives and function analysis |
| `calculus_optimisation_triplet_births.py` | Multiple Births | Gradient descent and curve fitting |

## References

Arden University (2026) *COM7023 Mathematics for Data Science Marking Matrix*. Arden University.

Human Multiple Births Database (2024) *FRA_InputData_25.11.2024.xlsx*. Available at: https://www.twinbirths.org/en/data-metadata/ (Accessed: 25 November 2024).

Stewart, J. (2015) *Calculus: Early Transcendentals*. 8th edn. Cengage Learning.
