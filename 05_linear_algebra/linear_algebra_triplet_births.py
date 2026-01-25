"""
================================================================================
LINEAR ALGEBRA - TRIPLET BIRTHS DATASET
================================================================================

Module Title:       Maths for Data Science
Module Code:        COM7023
Assignment Title:   Maths for Data Science
Student Number:     24154844
Student Name:       Farid Negahbnai
Tutor Name:         Ali Vaisifard
University:         Arden University

--------------------------------------------------------------------------------
DESCRIPTION:
This script applies linear algebra concepts to the triplet births demographic
dataset. We construct matrices from time series data and apply matrix
operations to analyse trends, correlations, and patterns in birth data.

MATHEMATICAL CONCEPTS:
- Vectors as time series
- Design matrices for regression
- Covariance and correlation matrices
- Eigenvalue decomposition for trend analysis
- Matrix projections

DATASET REFERENCE:
Human Multiple Births Database (2024) FRA_InputData_25.11.2024.xlsx.
Available at: https://www.twinbirths.org/en/data-metadata/
(Accessed: 25 November 2024).
--------------------------------------------------------------------------------
"""

# =============================================================================
# IMPORT REQUIRED LIBRARIES
# =============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# HEADER INFORMATION
# =============================================================================

print("=" * 70)
print("LINEAR ALGEBRA")
print("Dataset: Triplet Births (France)")
print("=" * 70)
print()
print("Module Title:       Maths for Data Science")
print("Module Code:        COM7023")
print("Student Number:     24154844")
print("Student Name:       Farid Negahbnai")
print("Tutor Name:         Ali Vaisifard")
print("University:         Arden University")
print("=" * 70)

# =============================================================================
# STEP 1: LOAD AND PREPARE THE DATASET
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING AND PREPARING THE DATASET")
print("=" * 70)

file_path = os.path.join(os.path.dirname(__file__), '../datasets/FRA_InputData_25.11.2024.xlsx')

try:
    df = pd.read_excel(file_path, sheet_name='input data')
    print("\nDataset loaded successfully!")
    print(f"Source: Human Multiple Births Database (2024)")
except FileNotFoundError:
    print("\nError: Dataset file not found.")
    exit()

# Data cleaning
df_clean = df.dropna(subset=['Year', 'Triplet_deliveries'])
df_clean['Year'] = df_clean['Year'].astype(int)

# Filter to analysis period
start_year = 1990
end_year = 2020
df_filtered = df_clean[(df_clean['Year'] >= start_year) & 
                        (df_clean['Year'] <= end_year)].copy()

# Extract relevant columns for analysis
if 'Twin_deliveries' in df_filtered.columns and 'Total_births' in df_filtered.columns:
    df_analysis = df_filtered[['Year', 'Total_births', 'Twin_deliveries', 
                               'Triplet_deliveries']].dropna()
else:
    # Create simulated additional columns if not available
    df_analysis = df_filtered[['Year', 'Triplet_deliveries']].copy()
    df_analysis['Total_births'] = 750000 + np.random.normal(0, 20000, len(df_analysis))
    df_analysis['Twin_deliveries'] = df_analysis['Triplet_deliveries'] * 50 + \
                                      np.random.normal(0, 100, len(df_analysis))

print(f"\nAnalysis Period: {start_year} - {end_year}")
print(f"Number of Observations: n = {len(df_analysis)} years")

print("\n--- Dataset Preview ---")
print(df_analysis.head(10).to_string(index=False))

# =============================================================================
# STEP 2: VECTORS AS TIME SERIES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: VECTORS AS TIME SERIES")
print("=" * 70)

print("\n--- Representing Time Series as Vectors ---")
print("\n  In linear algebra, a time series can be represented as a vector:")
print("\n       ┌ x₁ ┐     ┌ x₁₉₉₀ ┐")
print("  x =  │ x₂ │  =  │ x₁₉₉₁ │   where xₜ is the value at time t")
print("       │ ⋮  │     │   ⋮   │")
print("       └ xₙ ┘     └ x₂₀₂₀ ┘")

# Create vectors
years = df_analysis['Year'].values
triplets = df_analysis['Triplet_deliveries'].values
twins = df_analysis['Twin_deliveries'].values
total_births = df_analysis['Total_births'].values

n = len(years)

print(f"\n--- Time Series Vectors ---")
print(f"\n  Years vector (t):")
print(f"  t = [{years[0]}, {years[1]}, ..., {years[-1]}]ᵀ  ∈ ℝ^{n}")

print(f"\n  Triplet deliveries vector (x):")
print(f"  x = [{triplets[0]:.0f}, {triplets[1]:.0f}, ..., {triplets[-1]:.0f}]ᵀ  ∈ ℝ^{n}")

print(f"\n  Twin deliveries vector (y):")
print(f"  y = [{twins[0]:.0f}, {twins[1]:.0f}, ..., {twins[-1]:.0f}]ᵀ  ∈ ℝ^{n}")

# =============================================================================
# STEP 3: VECTOR OPERATIONS ON TIME SERIES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: VECTOR OPERATIONS ON TIME SERIES")
print("=" * 70)

print("\n" + "-" * 50)
print("3.1 MEAN VECTOR (CENTERING)")
print("-" * 50)

print("\n  Mean-centering: x̃ = x - μx1")
print("  where 1 is a vector of ones")

mean_triplets = np.mean(triplets)
mean_twins = np.mean(twins)

triplets_centered = triplets - mean_triplets
twins_centered = twins - mean_twins

print(f"\n  Mean of triplets: μₓ = {mean_triplets:.2f}")
print(f"  Mean of twins: μᵧ = {mean_twins:.2f}")

print(f"\n  Centered triplet vector (first 5 values):")
print(f"  x̃ = [{', '.join([f'{v:.2f}' for v in triplets_centered[:5]])}...]")

print(f"\n  Verification: mean(x̃) = {np.mean(triplets_centered):.10f} ≈ 0")

print("\n" + "-" * 50)
print("3.2 DOT PRODUCT AND CORRELATION")
print("-" * 50)

print("\n  The dot product of centered vectors relates to covariance:")
print("\n  Cov(X,Y) = (x̃ · ỹ) / (n-1) = (Σ x̃ᵢỹᵢ) / (n-1)")

# Calculate dot product
dot_product = np.dot(triplets_centered, twins_centered)
covariance = dot_product / (n - 1)

print(f"\n  x̃ · ỹ = Σ x̃ᵢỹᵢ = {dot_product:.2f}")
print(f"  Cov(X,Y) = {dot_product:.2f} / {n-1} = {covariance:.4f}")

print("\n--- Correlation Coefficient ---")
print("\n  Pearson correlation:")
print("  r = (x̃ · ỹ) / (||x̃|| × ||ỹ||)")

norm_triplets = np.linalg.norm(triplets_centered)
norm_twins = np.linalg.norm(twins_centered)
correlation = dot_product / (norm_triplets * norm_twins)

print(f"\n  ||x̃|| = {norm_triplets:.4f}")
print(f"  ||ỹ|| = {norm_twins:.4f}")
print(f"  r = {dot_product:.2f} / ({norm_triplets:.4f} × {norm_twins:.4f})")
print(f"  r = {correlation:.6f}")

# Verify with numpy
np_corr = np.corrcoef(triplets, twins)[0, 1]
print(f"\n  Verification (numpy): r = {np_corr:.6f}")

print(f"\n  Interpretation: {'Strong' if abs(correlation) > 0.7 else 'Moderate' if abs(correlation) > 0.4 else 'Weak'} ",
      f"{'positive' if correlation > 0 else 'negative'} correlation")

# =============================================================================
# STEP 4: DESIGN MATRIX FOR LINEAR REGRESSION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: DESIGN MATRIX FOR LINEAR REGRESSION")
print("=" * 70)

print("\n--- Linear Regression Model ---")
print("\n  Model: y = Xβ + ε")
print("\n  where:")
print("  - y is the response vector (n × 1)")
print("  - X is the design matrix (n × p)")
print("  - β is the coefficient vector (p × 1)")
print("  - ε is the error vector (n × 1)")

print("\n--- Constructing the Design Matrix ---")
print("\n  For simple linear regression with intercept:")
print("\n       ┌ 1   x₁ ┐")
print("  X =  │ 1   x₂ │")
print("       │ ⋮    ⋮ │")
print("       └ 1   xₙ ┘")

# Normalize years for numerical stability
years_normalized = (years - years.mean())

# Create design matrix
ones = np.ones(n)
X = np.column_stack([ones, years_normalized])

print(f"\n  Design matrix X (first 5 rows):")
print(f"  Shape: {X.shape} (n × p where p=2)")
print(f"\n  X = ")
for i in range(min(5, n)):
    print(f"    [{X[i, 0]:.0f}  {X[i, 1]:>8.4f}]")
print("    ...")

# =============================================================================
# STEP 5: SOLVING NORMAL EQUATIONS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: SOLVING NORMAL EQUATIONS")
print("=" * 70)

print("\n--- Normal Equations ---")
print("\n  The least squares solution minimizes ||y - Xβ||²")
print("\n  Solution: β = (XᵀX)⁻¹Xᵀy")
print("\n  Normal equations: XᵀXβ = Xᵀy")

# Response vector
y = triplets

# Step 1: Compute XᵀX
XtX = X.T @ X

print(f"\n  Step 1: Compute XᵀX")
print(f"  XᵀX = ")
print(f"    [{XtX[0, 0]:>10.4f}  {XtX[0, 1]:>10.4f}]")
print(f"    [{XtX[1, 0]:>10.4f}  {XtX[1, 1]:>10.4f}]")

# Step 2: Compute Xᵀy
Xty = X.T @ y

print(f"\n  Step 2: Compute Xᵀy")
print(f"  Xᵀy = ")
print(f"    [{Xty[0]:>10.4f}]")
print(f"    [{Xty[1]:>10.4f}]")

# Step 3: Solve for β
print(f"\n  Step 3: Solve (XᵀX)β = Xᵀy for β")

# Calculate determinant
det_XtX = np.linalg.det(XtX)
print(f"  det(XᵀX) = {det_XtX:.4f}")

# Calculate inverse
XtX_inv = np.linalg.inv(XtX)
print(f"\n  (XᵀX)⁻¹ = ")
print(f"    [{XtX_inv[0, 0]:>12.8f}  {XtX_inv[0, 1]:>12.8f}]")
print(f"    [{XtX_inv[1, 0]:>12.8f}  {XtX_inv[1, 1]:>12.8f}]")

# Calculate beta
beta = XtX_inv @ Xty

print(f"\n  β = (XᵀX)⁻¹Xᵀy = ")
print(f"    [β₀]   [{beta[0]:>10.4f}]   (intercept)")
print(f"    [β₁] = [{beta[1]:>10.4f}]   (slope)")

# Convert slope to original year scale
beta1_original = beta[1]
beta0_original = beta[0] - beta[1] * (-years.mean())

print(f"\n--- Regression Equation ---")
print(f"\n  ŷ = {beta[0]:.4f} + {beta[1]:.4f} × (year - {years.mean():.0f})")
print(f"\n  In original units:")
print(f"  ŷ = {beta0_original:.4f} + {beta1_original:.4f} × year")

# =============================================================================
# STEP 6: PROJECTION MATRIX (HAT MATRIX)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: PROJECTION MATRIX (HAT MATRIX)")
print("=" * 70)

print("\n--- Hat Matrix Definition ---")
print("\n  H = X(XᵀX)⁻¹Xᵀ")
print("\n  Properties:")
print("  - H projects y onto the column space of X")
print("  - ŷ = Hy (fitted values)")
print("  - H is symmetric: Hᵀ = H")
print("  - H is idempotent: H² = H")
print("  - trace(H) = p (number of parameters)")

# Calculate hat matrix
H = X @ XtX_inv @ X.T

print(f"\n  Shape of H: {H.shape}")
print(f"\n  Hat matrix H (first 5×5 block):")
for i in range(min(5, n)):
    row_str = '  '.join([f'{H[i, j]:.4f}' for j in range(min(5, n))])
    print(f"    [{row_str}]")
print("    ...")

# Verify properties
print(f"\n--- Verifying Hat Matrix Properties ---")

# Symmetry
is_symmetric = np.allclose(H, H.T)
print(f"\n  Symmetric (H = Hᵀ): {is_symmetric}")

# Idempotent
H_squared = H @ H
is_idempotent = np.allclose(H, H_squared)
print(f"  Idempotent (H² = H): {is_idempotent}")

# Trace
trace_H = np.trace(H)
print(f"  trace(H) = {trace_H:.4f} (should equal p = {X.shape[1]})")

# Fitted values
y_hat = H @ y
print(f"\n--- Fitted Values: ŷ = Hy ---")
print(f"\n  Year    Actual    Fitted    Residual")
print("  " + "-" * 45)
for i in range(min(10, n)):
    residual = y[i] - y_hat[i]
    print(f"  {int(years[i])}    {y[i]:>6.0f}    {y_hat[i]:>6.1f}    {residual:>+7.1f}")
print("  ...")

# =============================================================================
# STEP 7: COVARIANCE MATRIX ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: COVARIANCE MATRIX ANALYSIS")
print("=" * 70)

print("\n--- Multi-variable Data Matrix ---")
print("\n  Create data matrix with multiple birth variables:")

# Create data matrix
data_matrix = np.column_stack([triplets, twins, total_births / 1000])  # Scale total births
column_names = ['Triplets', 'Twins', 'Total (k)']

print(f"\n  Data matrix Z (first 5 rows):")
print(f"  Shape: {data_matrix.shape}")
print(f"\n  {'Year':>6}  {'Triplets':>10}  {'Twins':>10}  {'Total(k)':>10}")
print("  " + "-" * 50)
for i in range(min(5, n)):
    print(f"  {int(years[i]):>6}  {data_matrix[i, 0]:>10.0f}  "
          f"{data_matrix[i, 1]:>10.0f}  {data_matrix[i, 2]:>10.1f}")
print("  ...")

print("\n--- Covariance Matrix Calculation ---")
print("\n  Formula: Σ = (1/(n-1)) × (Z - Z̄)ᵀ(Z - Z̄)")

# Center the data
Z_centered = data_matrix - np.mean(data_matrix, axis=0)

# Calculate covariance matrix manually
cov_matrix_manual = (Z_centered.T @ Z_centered) / (n - 1)

# Verify with numpy
cov_matrix_numpy = np.cov(data_matrix, rowvar=False)

print(f"\n  Covariance Matrix Σ:")
print(f"\n  {'':>12}  {'Triplets':>12}  {'Twins':>12}  {'Total(k)':>12}")
print("  " + "-" * 55)
for i, name in enumerate(column_names):
    row_str = '  '.join([f'{cov_matrix_numpy[i, j]:>12.2f}' for j in range(3)])
    print(f"  {name:>12}  {row_str}")

print("\n--- Interpretation ---")
print(f"\n  Variance(Triplets) = {cov_matrix_numpy[0, 0]:.2f}")
print(f"  Variance(Twins) = {cov_matrix_numpy[1, 1]:.2f}")
print(f"  Cov(Triplets, Twins) = {cov_matrix_numpy[0, 1]:.2f}")

# =============================================================================
# STEP 8: CORRELATION MATRIX
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: CORRELATION MATRIX")
print("=" * 70)

print("\n--- Correlation Matrix Calculation ---")
print("\n  Formula: R = D⁻¹ΣD⁻¹")
print("  where D = diag(√σ₁₁, √σ₂₂, ...)")

# Calculate correlation matrix
std_devs = np.sqrt(np.diag(cov_matrix_numpy))
D_inv = np.diag(1 / std_devs)
corr_matrix = D_inv @ cov_matrix_numpy @ D_inv

# Verify with numpy
corr_matrix_numpy = np.corrcoef(data_matrix, rowvar=False)

print(f"\n  Correlation Matrix R:")
print(f"\n  {'':>12}  {'Triplets':>12}  {'Twins':>12}  {'Total(k)':>12}")
print("  " + "-" * 55)
for i, name in enumerate(column_names):
    row_str = '  '.join([f'{corr_matrix_numpy[i, j]:>12.4f}' for j in range(3)])
    print(f"  {name:>12}  {row_str}")

print("\n--- Interpretation ---")
for i in range(3):
    for j in range(i+1, 3):
        r = corr_matrix_numpy[i, j]
        strength = 'Strong' if abs(r) > 0.7 else 'Moderate' if abs(r) > 0.4 else 'Weak'
        direction = 'positive' if r > 0 else 'negative'
        print(f"  Corr({column_names[i]}, {column_names[j]}) = {r:.4f} [{strength} {direction}]")

# =============================================================================
# STEP 9: EIGENVALUE DECOMPOSITION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: EIGENVALUE DECOMPOSITION")
print("=" * 70)

print("\n--- Eigendecomposition of Covariance Matrix ---")
print("\n  Σ = VΛVᵀ")
print("  where:")
print("  - V contains eigenvectors (columns)")
print("  - Λ is diagonal matrix of eigenvalues")

# Compute eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix_numpy)

# Sort by eigenvalue (descending)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\n--- Eigenvalues ---")
print(f"\n  Eigenvalues (sorted descending):")
for i, ev in enumerate(eigenvalues):
    print(f"    λ{i+1} = {ev:.4f}")

print(f"\n  Sum of eigenvalues = {sum(eigenvalues):.4f}")
print(f"  trace(Σ) = {np.trace(cov_matrix_numpy):.4f}")
print(f"  (These should be equal: {np.isclose(sum(eigenvalues), np.trace(cov_matrix_numpy))})")

print(f"\n--- Eigenvectors (Principal Components) ---")
print(f"\n  {'':>12}  {'PC1':>12}  {'PC2':>12}  {'PC3':>12}")
print("  " + "-" * 55)
for i, name in enumerate(column_names):
    row_str = '  '.join([f'{eigenvectors[i, j]:>12.4f}' for j in range(3)])
    print(f"  {name:>12}  {row_str}")

print("\n--- Variance Explained ---")
total_variance = sum(eigenvalues)
for i, ev in enumerate(eigenvalues):
    pct = (ev / total_variance) * 100
    cumulative = sum(eigenvalues[:i+1]) / total_variance * 100
    print(f"  PC{i+1}: {pct:.2f}% (cumulative: {cumulative:.2f}%)")

print("\n--- Interpretation ---")
print("\n  Principal Component 1 captures the dominant direction of variation")
print("  in the data. This is used in PCA for dimensionality reduction.")

# =============================================================================
# STEP 10: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Time series vectors
ax1 = axes[0, 0]
ax1.plot(years, triplets, 'o-', linewidth=2, markersize=5, 
         label='Triplets', color='steelblue')
ax1.plot(years, twins / 100, 's--', linewidth=2, markersize=4, 
         label='Twins (÷100)', color='green', alpha=0.7)
ax1.axhline(mean_triplets, color='steelblue', linestyle=':', alpha=0.5)
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Number of Deliveries', fontsize=11)
ax1.set_title('Time Series Vectors\n(Triplets and Twins)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Linear regression
ax2 = axes[0, 1]
ax2.scatter(years, triplets, color='steelblue', s=50, alpha=0.7, 
            edgecolors='white', label='Actual Data')
ax2.plot(years, y_hat, 'r-', linewidth=2.5, label='ŷ = Xβ')
ax2.fill_between(years, y_hat - np.std(y - y_hat), y_hat + np.std(y - y_hat),
                 alpha=0.2, color='red', label='±1 Std Error')
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Triplet Deliveries', fontsize=11)
ax2.set_title('Linear Regression: ŷ = Xβ\nβ = (XᵀX)⁻¹Xᵀy', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Add equation
eq_text = f'ŷ = {beta[0]:.1f} + {beta[1]:.2f}(year - {years.mean():.0f})'
ax2.text(0.05, 0.95, eq_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))

# Plot 3: Covariance/Correlation heatmap
ax3 = axes[1, 0]
im = ax3.imshow(corr_matrix_numpy, cmap='RdBu_r', vmin=-1, vmax=1)
ax3.set_xticks(range(3))
ax3.set_xticklabels(column_names, fontsize=10)
ax3.set_yticks(range(3))
ax3.set_yticklabels(column_names, fontsize=10)
ax3.set_title('Correlation Matrix R\nR = D⁻¹ΣD⁻¹', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax3, label='Correlation')

# Add text annotations
for i in range(3):
    for j in range(3):
        color = 'white' if abs(corr_matrix_numpy[i, j]) > 0.5 else 'black'
        ax3.text(j, i, f'{corr_matrix_numpy[i, j]:.3f}', 
                ha='center', va='center', color=color, fontsize=11)

# Plot 4: Eigenvalue spectrum and PCA
ax4 = axes[1, 1]

# Bar chart of variance explained
variance_explained = eigenvalues / sum(eigenvalues) * 100
cumulative_variance = np.cumsum(variance_explained)

bars = ax4.bar(range(len(eigenvalues)), variance_explained, 
               color='steelblue', edgecolor='white', alpha=0.8, 
               label='Individual')
ax4.set_xticks(range(len(eigenvalues)))
ax4.set_xticklabels([f'PC{i+1}' for i in range(len(eigenvalues))])
ax4.set_xlabel('Principal Component', fontsize=11)
ax4.set_ylabel('Variance Explained (%)', fontsize=11)
ax4.set_title('Eigenvalue Decomposition: Σ = VΛVᵀ\n(Variance Explained by Each PC)', 
              fontsize=12, fontweight='bold')

# Add cumulative line
ax4_twin = ax4.twinx()
ax4_twin.plot(range(len(eigenvalues)), cumulative_variance, 'ro-', 
              linewidth=2, markersize=8, label='Cumulative')
ax4_twin.set_ylabel('Cumulative Variance (%)', color='red', fontsize=11)
ax4_twin.tick_params(axis='y', labelcolor='red')
ax4_twin.set_ylim(0, 105)

# Add eigenvalue labels
for i, (bar, ev) in enumerate(zip(bars, eigenvalues)):
    height = bar.get_height()
    ax4.annotate(f'λ={ev:.1f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=9)

ax4.legend(loc='upper left')
ax4_twin.legend(loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), '../outputs/figures/linear_algebra_triplet_births.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: linear_algebra_triplet_births.png")

# =============================================================================
# STEP 11: APPLICATIONS SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 11: APPLICATIONS IN DATA SCIENCE")
print("=" * 70)

print("\n--- 1. Linear Regression ---")
print(f"\n  Normal equations: β = (XᵀX)⁻¹Xᵀy")
print(f"  Estimated coefficients:")
print(f"    β₀ (intercept) = {beta[0]:.4f}")
print(f"    β₁ (slope)     = {beta[1]:.4f}")

# Calculate R-squared
SS_res = np.sum((y - y_hat) ** 2)
SS_tot = np.sum((y - np.mean(y)) ** 2)
R_squared = 1 - (SS_res / SS_tot)
print(f"  R² = 1 - SS_res/SS_tot = {R_squared:.4f}")

print("\n--- 2. Principal Component Analysis ---")
print(f"\n  PC1 explains {variance_explained[0]:.1f}% of total variance")
print(f"  PC1 + PC2 explain {cumulative_variance[1]:.1f}% of total variance")

print("\n--- 3. Covariance Analysis ---")
print(f"\n  Strongest correlation: Triplets-Twins (r = {corr_matrix_numpy[0, 1]:.4f})")
print(f"  This suggests related demographic trends")

# =============================================================================
# STEP 12: SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 12: SUMMARY")
print("=" * 70)

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│                   LINEAR ALGEBRA SUMMARY                             │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│  Time Series Length:         n = {n:>4} years                         │")
print(f"│  Design Matrix Shape:        X = {X.shape[0]:>3} × {X.shape[1]}                           │")
print(f"│  Covariance Matrix Shape:    Σ = {cov_matrix_numpy.shape[0]:>3} × {cov_matrix_numpy.shape[1]}                           │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│  Regression R²:              {R_squared:>8.4f}                             │")
print(f"│  Slope (β₁):                 {beta[1]:>8.4f}                             │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│  Dominant Eigenvalue (λ₁):   {eigenvalues[0]:>8.2f}                             │")
print(f"│  Variance by PC1:            {variance_explained[0]:>7.2f}%                             │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\n--- Dataset Reference ---")
print("  Human Multiple Births Database (2024) FRA_InputData_25.11.2024.xlsx.")
print("  Available at: https://www.twinbirths.org/en/data-metadata/")
print("  (Accessed: 25 November 2024).")

print("\n--- Further Reading ---")
print("  Strang, G. (2016) Introduction to Linear Algebra. 5th edn.")
print("  Wellesley-Cambridge Press.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF LINEAR ALGEBRA ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
