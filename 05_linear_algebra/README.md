# Linear Algebra

## Overview

Linear algebra forms the mathematical foundation of modern data science, machine learning, and artificial intelligence. This section demonstrates fundamental linear algebra concepts including vectors, matrices, and their operations.

## Mathematical Concepts

### Vectors

A vector is an ordered list of numbers representing a point or direction in space:

$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$$

**Vector Operations:**

- Addition: $\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$

- Dot Product: $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i$

- Magnitude: $\|\mathbf{v}\| = \sqrt{\sum_{i=1}^{n} v_i^2}$

### Matrices

A matrix is a rectangular array of numbers:

$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

**Matrix Operations:**

- Transpose: $(\mathbf{A}^T)_{ij} = a_{ji}$

- Multiplication: $(\mathbf{AB})_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj}$

- Determinant (2×2): $\det(\mathbf{A}) = a_{11}a_{22} - a_{12}a_{21}$

### Eigenvalues and Eigenvectors

For a square matrix $\mathbf{A}$, eigenvalue $\lambda$ and eigenvector $\mathbf{v}$ satisfy:

$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$

## Applications in Data Science

1. **Principal Component Analysis (PCA)**: Uses eigendecomposition for dimensionality reduction
2. **Linear Regression**: Solves $\mathbf{X}^T\mathbf{X}\boldsymbol{\beta} = \mathbf{X}^T\mathbf{y}$
3. **Covariance Matrices**: Capture relationships between variables
4. **Neural Networks**: Weight matrices transform inputs

## Scripts

| Script | Dataset | Description |
|--------|---------|-------------|
| `linear_algebra_marking_matrix.py` | Marking Matrix | Matrix representation and operations |
| `linear_algebra_triplet_births.py` | Multiple Births | Covariance matrices and eigenanalysis |

## References

Arden University (2026) *COM7023 Mathematics for Data Science Marking Matrix*. Arden University.

Human Multiple Births Database (2024) *FRA_InputData_25.11.2024.xlsx*. Available at: https://www.twinbirths.org/en/data-metadata/ (Accessed: 25 November 2024).

Strang, G. (2016) *Introduction to Linear Algebra*. 5th edn. Wellesley-Cambridge Press.
