"""
================================================================================
CORRELATION AND REGRESSION - MARKING MATRIX DATASET
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
This script demonstrates correlation and regression analysis using the marking
matrix dataset. We explore relationships between grade bands and learning
outcomes, calculate correlation coefficients, and build regression models.

MATHEMATICAL CONCEPTS:
- Pearson correlation coefficient
- Spearman rank correlation
- Simple linear regression
- Multiple regression
- Coefficient of determination (R²)
- Residual analysis

DATASET REFERENCE:
Arden University (2024) COM7023 Mathematics for Data Science Marking Matrix.
Arden University.
--------------------------------------------------------------------------------
"""

# =============================================================================
# IMPORT REQUIRED LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

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
print("CORRELATION AND REGRESSION")
print("Dataset: Marking Matrix")
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

file_path = '../datasets/COM7023_Mathematics_for_Data_Science_Marking_Matrix.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("\nDataset loaded successfully!")
except FileNotFoundError:
    print("\nError: Dataset file not found.")
    exit()

# Define grade columns and midpoints
grade_columns = ['90- 100', '80 - 90', '70 – 79', '60 - 69', 
                 '50 – 59', '40 – 49', '30 – 39', '20 - 29', '0 - 19']

grade_midpoints = {
    '90- 100': 95, '80 - 90': 85, '70 – 79': 74.5, '60 - 69': 64.5,
    '50 – 59': 54.5, '40 – 49': 44.5, '30 – 39': 34.5, '20 - 29': 24.5, '0 - 19': 9.5
}

# Create numerical features matrix (word counts)
feature_data = []
for idx, row in df.iterrows():
    lo_name = row['Learning Outcome'] if 'Learning Outcome' in df.columns else f'LO{idx+1}'
    row_data = {'LO': lo_name}
    for grade_col in grade_columns:
        if grade_col in df.columns:
            text = str(row[grade_col])
            word_count = len(text.split())
            row_data[f'Grade_{grade_midpoints[grade_col]}'] = word_count
    feature_data.append(row_data)

df_features = pd.DataFrame(feature_data)
feature_cols = [col for col in df_features.columns if col.startswith('Grade_')]

print(f"\nNumber of Learning Outcomes: {len(df_features)}")
print(f"Number of Grade Bands: {len(feature_cols)}")

print("\n--- Feature Matrix (Word Counts) ---")
print(df_features.head().to_string(index=False))

# =============================================================================
# STEP 2: PEARSON CORRELATION COEFFICIENT
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: PEARSON CORRELATION COEFFICIENT")
print("=" * 70)

print("\n--- Definition ---")
print("\n  The Pearson correlation coefficient measures linear association:")
print("\n            Σ(xᵢ - x̄)(yᵢ - ȳ)")
print("  r = ─────────────────────────────────")
print("       √[Σ(xᵢ - x̄)²] × √[Σ(yᵢ - ȳ)²]")
print("\n  Properties:")
print("    -1 ≤ r ≤ 1")
print("    r = 1:  Perfect positive linear relationship")
print("    r = -1: Perfect negative linear relationship")
print("    r = 0:  No linear relationship")

print("\n--- Manual Calculation Example ---")

# Select two grade bands for demonstration
x = df_features['Grade_95'].values  # Highest grade band
y = df_features['Grade_54.5'].values  # Middle grade band

n = len(x)
x_mean = np.mean(x)
y_mean = np.mean(y)

# Calculate components
numerator = np.sum((x - x_mean) * (y - y_mean))
denominator_x = np.sqrt(np.sum((x - x_mean) ** 2))
denominator_y = np.sqrt(np.sum((y - y_mean) ** 2))
r_manual = numerator / (denominator_x * denominator_y)

print(f"\n  Variables: Grade_95 (x) vs Grade_54.5 (y)")
print(f"\n  Step 1: Calculate means")
print(f"    x̄ = {x_mean:.4f}")
print(f"    ȳ = {y_mean:.4f}")

print(f"\n  Step 2: Calculate deviations")
print("    i    xᵢ     yᵢ    (xᵢ-x̄)   (yᵢ-ȳ)  (xᵢ-x̄)(yᵢ-ȳ)")
print("    " + "-" * 55)
for i in range(min(4, n)):
    dev_x = x[i] - x_mean
    dev_y = y[i] - y_mean
    product = dev_x * dev_y
    print(f"    {i+1}    {x[i]:>4}   {y[i]:>4}   {dev_x:>7.2f}  {dev_y:>7.2f}    {product:>8.2f}")

print(f"\n  Step 3: Calculate correlation")
print(f"    Σ(xᵢ - x̄)(yᵢ - ȳ) = {numerator:.4f}")
print(f"    √Σ(xᵢ - x̄)² = {denominator_x:.4f}")
print(f"    √Σ(yᵢ - ȳ)² = {denominator_y:.4f}")
print(f"    r = {numerator:.4f} / ({denominator_x:.4f} × {denominator_y:.4f})")
print(f"    r = {r_manual:.6f}")

# Verify with scipy
r_scipy, p_value = stats.pearsonr(x, y)
print(f"\n  Verification (scipy.stats.pearsonr):")
print(f"    r = {r_scipy:.6f}")
print(f"    p-value = {p_value:.6f}")

# =============================================================================
# STEP 3: CORRELATION MATRIX
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: CORRELATION MATRIX")
print("=" * 70)

print("\n--- Computing Correlation Matrix ---")
print("\n  A correlation matrix shows pairwise correlations between all variables.")

# Calculate correlation matrix
corr_matrix = df_features[feature_cols].corr()

print("\n  Correlation Matrix:")
print("  " + "-" * 80)

# Create shortened column names for display
short_names = {col: f'G{int(float(col.split("_")[1]))}' for col in feature_cols}

# Display correlation matrix
header = "        " + "  ".join([f"{short_names[col]:>6}" for col in feature_cols])
print(header)
for i, row_col in enumerate(feature_cols):
    row_values = "  ".join([f"{corr_matrix.loc[row_col, col]:>6.3f}" for col in feature_cols])
    print(f"  {short_names[row_col]:>4}  {row_values}")

print("\n--- Interpretation ---")

# Find strongest positive and negative correlations (excluding diagonal)
corr_pairs = []
for i, col1 in enumerate(feature_cols):
    for j, col2 in enumerate(feature_cols):
        if i < j:  # Upper triangle only
            corr_pairs.append((col1, col2, corr_matrix.loc[col1, col2]))

corr_pairs.sort(key=lambda x: x[2])

print("\n  Strongest Positive Correlations:")
for col1, col2, r in corr_pairs[-3:]:
    print(f"    {short_names[col1]} vs {short_names[col2]}: r = {r:.4f}")

print("\n  Weakest/Negative Correlations:")
for col1, col2, r in corr_pairs[:3]:
    print(f"    {short_names[col1]} vs {short_names[col2]}: r = {r:.4f}")

# =============================================================================
# STEP 4: SPEARMAN RANK CORRELATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: SPEARMAN RANK CORRELATION")
print("=" * 70)

print("\n--- Definition ---")
print("\n  Spearman's rank correlation measures monotonic relationships:")
print("\n  1. Rank each variable from 1 to n")
print("  2. Calculate Pearson correlation on ranks")
print("\n  Formula (when no ties):")
print("              6 × Σdᵢ²")
print("  ρ = 1 - ─────────────")
print("           n(n² - 1)")
print("\n  where dᵢ = rank(xᵢ) - rank(yᵢ)")

print("\n--- Example Calculation ---")

# Use same variables
x_ranks = stats.rankdata(x)
y_ranks = stats.rankdata(y)

print(f"\n  Original and Ranked Values:")
print("    i    xᵢ   rank(xᵢ)    yᵢ   rank(yᵢ)    dᵢ")
print("    " + "-" * 50)
d_squared_sum = 0
for i in range(min(4, n)):
    d = x_ranks[i] - y_ranks[i]
    d_squared_sum += d ** 2
    print(f"    {i+1}    {x[i]:>3}      {x_ranks[i]:>4.1f}    {y[i]:>3}      {y_ranks[i]:>4.1f}   {d:>6.1f}")

# Calculate Spearman using formula
total_d_squared = np.sum((x_ranks - y_ranks) ** 2)
rho_formula = 1 - (6 * total_d_squared) / (n * (n**2 - 1))

# Verify with scipy
rho_scipy, p_spearman = stats.spearmanr(x, y)

print(f"\n  Calculation:")
print(f"    Σdᵢ² = {total_d_squared:.2f}")
print(f"    n = {n}")
print(f"    ρ = 1 - (6 × {total_d_squared:.2f}) / ({n} × ({n}² - 1))")
print(f"    ρ = 1 - {6 * total_d_squared:.2f} / {n * (n**2 - 1)}")
print(f"    ρ = {rho_formula:.6f}")

print(f"\n  Verification (scipy.stats.spearmanr):")
print(f"    ρ = {rho_scipy:.6f}")
print(f"    p-value = {p_spearman:.6f}")

print(f"\n--- Comparison: Pearson vs Spearman ---")
print(f"    Pearson r = {r_scipy:.6f}")
print(f"    Spearman ρ = {rho_scipy:.6f}")
print(f"\n  Interpretation:")
if abs(r_scipy - rho_scipy) < 0.1:
    print("    Similar values suggest a linear relationship")
elif abs(rho_scipy) > abs(r_scipy):
    print("    Higher Spearman suggests a monotonic but non-linear relationship")

# =============================================================================
# STEP 5: SIMPLE LINEAR REGRESSION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: SIMPLE LINEAR REGRESSION")
print("=" * 70)

print("\n--- The Linear Regression Model ---")
print("\n  Model: y = β₀ + β₁x + ε")
print("\n  where:")
print("    β₀ = intercept (y-value when x = 0)")
print("    β₁ = slope (change in y per unit change in x)")
print("    ε = random error term")

print("\n--- Least Squares Estimation ---")
print("\n  Minimise the sum of squared residuals:")
print("    SSE = Σ(yᵢ - ŷᵢ)² = Σ(yᵢ - β₀ - β₁xᵢ)²")
print("\n  Solutions:")
print("           Σ(xᵢ - x̄)(yᵢ - ȳ)")
print("    β̂₁ = ─────────────────────")
print("             Σ(xᵢ - x̄)²")
print("\n    β̂₀ = ȳ - β̂₁x̄")

# Calculate regression coefficients manually
beta_1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
beta_0 = y_mean - beta_1 * x_mean

print(f"\n--- Calculation (Grade_95 predicting Grade_54.5) ---")
print(f"\n  Step 1: Calculate means")
print(f"    x̄ = {x_mean:.4f}")
print(f"    ȳ = {y_mean:.4f}")

print(f"\n  Step 2: Calculate β̂₁ (slope)")
numerator_beta = np.sum((x - x_mean) * (y - y_mean))
denominator_beta = np.sum((x - x_mean) ** 2)
print(f"    Σ(xᵢ - x̄)(yᵢ - ȳ) = {numerator_beta:.4f}")
print(f"    Σ(xᵢ - x̄)² = {denominator_beta:.4f}")
print(f"    β̂₁ = {numerator_beta:.4f} / {denominator_beta:.4f}")
print(f"    β̂₁ = {beta_1:.6f}")

print(f"\n  Step 3: Calculate β̂₀ (intercept)")
print(f"    β̂₀ = {y_mean:.4f} - ({beta_1:.6f} × {x_mean:.4f})")
print(f"    β̂₀ = {beta_0:.6f}")

print(f"\n  Regression Equation:")
print(f"    ŷ = {beta_0:.4f} + {beta_1:.4f}x")

# Verify with scipy
slope_scipy, intercept_scipy, r_value, p_value, std_err = stats.linregress(x, y)
print(f"\n  Verification (scipy.stats.linregress):")
print(f"    β̂₁ = {slope_scipy:.6f}")
print(f"    β̂₀ = {intercept_scipy:.6f}")

# =============================================================================
# STEP 6: COEFFICIENT OF DETERMINATION (R²)
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: COEFFICIENT OF DETERMINATION (R²)")
print("=" * 70)

print("\n--- Definition ---")
print("\n  R² measures the proportion of variance in y explained by x:")
print("\n           SSₜₒₜ - SSᵣₑₛ       SSᵣₑₛ")
print("  R² = ─────────────── = 1 - ─────────")
print("            SSₜₒₜ             SSₜₒₜ")
print("\n  where:")
print("    SSₜₒₜ = Σ(yᵢ - ȳ)²     (Total Sum of Squares)")
print("    SSᵣₑₛ = Σ(yᵢ - ŷᵢ)²   (Residual Sum of Squares)")

# Calculate predictions
y_pred = beta_0 + beta_1 * x

# Calculate sums of squares
SS_tot = np.sum((y - y_mean) ** 2)
SS_res = np.sum((y - y_pred) ** 2)
SS_reg = np.sum((y_pred - y_mean) ** 2)

R_squared = 1 - (SS_res / SS_tot)

print(f"\n--- Calculation ---")
print(f"\n  Step 1: Calculate predicted values")
print("    i    yᵢ     ŷᵢ    (yᵢ - ȳ)²  (yᵢ - ŷᵢ)²")
print("    " + "-" * 50)
for i in range(min(4, n)):
    ss_tot_i = (y[i] - y_mean) ** 2
    ss_res_i = (y[i] - y_pred[i]) ** 2
    print(f"    {i+1}   {y[i]:>4}  {y_pred[i]:>6.2f}   {ss_tot_i:>8.2f}   {ss_res_i:>8.2f}")

print(f"\n  Step 2: Calculate sums of squares")
print(f"    SSₜₒₜ = Σ(yᵢ - ȳ)² = {SS_tot:.4f}")
print(f"    SSᵣₑₛ = Σ(yᵢ - ŷᵢ)² = {SS_res:.4f}")
print(f"    SSᵣₑᵍ = Σ(ŷᵢ - ȳ)² = {SS_reg:.4f}")
print(f"\n    Verification: SSₜₒₜ ≈ SSᵣₑₛ + SSᵣₑᵍ")
print(f"    {SS_tot:.4f} ≈ {SS_res:.4f} + {SS_reg:.4f} = {SS_res + SS_reg:.4f}")

print(f"\n  Step 3: Calculate R²")
print(f"    R² = 1 - ({SS_res:.4f} / {SS_tot:.4f})")
print(f"    R² = 1 - {SS_res/SS_tot:.6f}")
print(f"    R² = {R_squared:.6f}")

print(f"\n--- Interpretation ---")
print(f"    {R_squared*100:.2f}% of the variance in Grade_54.5 is explained by Grade_95")
print(f"\n    Note: R² = r² for simple linear regression")
print(f"    r² = {r_scipy**2:.6f} ≈ R² = {R_squared:.6f}")

# =============================================================================
# STEP 7: RESIDUAL ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: RESIDUAL ANALYSIS")
print("=" * 70)

print("\n--- Residuals ---")
print("\n  Residuals are the differences between observed and predicted values:")
print("    eᵢ = yᵢ - ŷᵢ")
print("\n  Properties of residuals in OLS:")
print("    1. Σeᵢ = 0 (residuals sum to zero)")
print("    2. Σxᵢeᵢ = 0 (residuals uncorrelated with x)")
print("    3. Mean of residuals = 0")

residuals = y - y_pred

print(f"\n--- Residual Summary ---")
print(f"    Sum of residuals: Σeᵢ = {np.sum(residuals):.10f} ≈ 0")
print(f"    Mean of residuals: ē = {np.mean(residuals):.10f} ≈ 0")
print(f"    Σxᵢeᵢ = {np.sum(x * residuals):.10f} ≈ 0")

print(f"\n--- Residual Statistics ---")
print(f"    Standard deviation of residuals: {np.std(residuals, ddof=2):.4f}")
print(f"    Minimum residual: {np.min(residuals):.4f}")
print(f"    Maximum residual: {np.max(residuals):.4f}")

# Standardised residuals
std_residuals = residuals / np.std(residuals, ddof=2)
print(f"\n--- Standardised Residuals ---")
print("    (Should be within ±2 for 95% of observations)")
for i, sr in enumerate(std_residuals):
    flag = "*" if abs(sr) > 2 else ""
    print(f"    Observation {i+1}: {sr:>7.4f} {flag}")

# =============================================================================
# STEP 8: SIGNIFICANCE TESTING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: SIGNIFICANCE TESTING")
print("=" * 70)

print("\n--- Hypothesis Test for Slope ---")
print("\n  H₀: β₁ = 0 (no linear relationship)")
print("  H₁: β₁ ≠ 0 (linear relationship exists)")
print("\n  Test statistic:")
print("           β̂₁")
print("    t = ────────")
print("         SE(β̂₁)")

# Calculate standard error of slope
MSE = SS_res / (n - 2)
SE_beta1 = np.sqrt(MSE / np.sum((x - x_mean) ** 2))

t_statistic = beta_1 / SE_beta1
df = n - 2
p_value_t = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

print(f"\n--- Calculation ---")
print(f"    MSE = SSᵣₑₛ / (n - 2) = {SS_res:.4f} / {n - 2} = {MSE:.4f}")
print(f"    SE(β̂₁) = √(MSE / Σ(xᵢ - x̄)²) = √({MSE:.4f} / {denominator_beta:.4f})")
print(f"    SE(β̂₁) = {SE_beta1:.6f}")
print(f"\n    t = {beta_1:.6f} / {SE_beta1:.6f} = {t_statistic:.4f}")
print(f"    Degrees of freedom: df = n - 2 = {df}")
print(f"    p-value (two-tailed) = {p_value_t:.6f}")

alpha = 0.05
print(f"\n--- Decision (α = {alpha}) ---")
if p_value_t < alpha:
    print(f"    p-value ({p_value_t:.6f}) < α ({alpha})")
    print("    Reject H₀: There is a significant linear relationship")
else:
    print(f"    p-value ({p_value_t:.6f}) ≥ α ({alpha})")
    print("    Fail to reject H₀: No significant linear relationship")

# =============================================================================
# STEP 9: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Correlation Matrix Heatmap
ax1 = axes[0, 0]
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, ax=ax1, cbar_kws={'shrink': 0.8},
            xticklabels=[short_names[c] for c in feature_cols],
            yticklabels=[short_names[c] for c in feature_cols])
ax1.set_title('Pearson Correlation Matrix\n(Grade Band Word Counts)', 
              fontsize=12, fontweight='bold')

# Plot 2: Scatter plot with regression line
ax2 = axes[0, 1]
ax2.scatter(x, y, color='steelblue', s=80, alpha=0.7, edgecolors='white',
           label='Observed Data')
x_line = np.linspace(min(x) - 1, max(x) + 1, 100)
y_line = beta_0 + beta_1 * x_line
ax2.plot(x_line, y_line, 'r-', linewidth=2, label=f'ŷ = {beta_0:.2f} + {beta_1:.2f}x')

# Add confidence band
se_pred = np.sqrt(MSE * (1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2)))
t_crit = stats.t.ppf(0.975, df)
ax2.fill_between(x_line, y_line - t_crit * se_pred, y_line + t_crit * se_pred,
                alpha=0.2, color='red', label='95% CI')

ax2.set_xlabel('Grade_95 (Word Count)', fontsize=11)
ax2.set_ylabel('Grade_54.5 (Word Count)', fontsize=11)
ax2.set_title(f'Simple Linear Regression\nR² = {R_squared:.4f}, r = {r_scipy:.4f}', 
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residual Plot
ax3 = axes[1, 0]
ax3.scatter(y_pred, residuals, color='steelblue', s=80, alpha=0.7, edgecolors='white')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.axhline(y=2*np.std(residuals), color='orange', linestyle=':', alpha=0.7)
ax3.axhline(y=-2*np.std(residuals), color='orange', linestyle=':', alpha=0.7)
ax3.set_xlabel('Predicted Values (ŷ)', fontsize=11)
ax3.set_ylabel('Residuals (e = y - ŷ)', fontsize=11)
ax3.set_title('Residual Plot\n(Check for Homoscedasticity)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Q-Q Plot for Normality
ax4 = axes[1, 1]
stats.probplot(residuals, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot of Residuals\n(Check for Normality)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/correlation_regression_marking_matrix.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: correlation_regression_marking_matrix.png")

# =============================================================================
# STEP 10: SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: SUMMARY")
print("=" * 70)

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│               CORRELATION AND REGRESSION SUMMARY                     │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  CORRELATION ANALYSIS                                                │")
print(f"│    Pearson r (Grade_95 vs Grade_54.5):  {r_scipy:>10.6f}                │")
print(f"│    Spearman ρ:                          {rho_scipy:>10.6f}                │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  REGRESSION ANALYSIS                                                 │")
print(f"│    Intercept (β̂₀):                      {beta_0:>10.4f}                    │")
print(f"│    Slope (β̂₁):                          {beta_1:>10.6f}                │")
print(f"│    R²:                                  {R_squared:>10.6f}                │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  HYPOTHESIS TEST                                                     │")
print(f"│    t-statistic:                         {t_statistic:>10.4f}                    │")
print(f"│    p-value:                             {p_value_t:>10.6f}                │")
print(f"│    Significant (α=0.05):                {'Yes' if p_value_t < 0.05 else 'No':>10}                    │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\n--- Dataset Reference ---")
print("  Arden University (2024) COM7023 Mathematics for Data Science")
print("  Marking Matrix. Arden University.")

print("\n--- Further Reading ---")
print("  Freedman, D.A. (2009) Statistical Models: Theory and Practice.")
print("  Cambridge University Press.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF CORRELATION AND REGRESSION ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
