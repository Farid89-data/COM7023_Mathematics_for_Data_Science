"""
================================================================================
CORRELATION AND REGRESSION - TRIPLET BIRTHS DATASET
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
This script applies correlation and regression analysis to the triplet births
dataset. We explore temporal trends, build predictive models, and assess
model quality using various diagnostic measures.

MATHEMATICAL CONCEPTS:
- Time series correlation
- Simple and polynomial regression
- Multiple regression
- Model comparison (R², AIC, BIC)
- Prediction intervals

DATASET REFERENCE:
Human Multiple Births Database (2024) FRA_InputData_25.11.2024.xlsx.
Available at: https://www.twinbirths.org/en/data-metadata/
(Accessed: 25 November 2024).
--------------------------------------------------------------------------------
"""

# =============================================================================
# IMPORT REQUIRED LIBRARIES
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

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

file_path = '../datasets/FRA_InputData_25.11.2024.xlsx'

try:
    df = pd.read_excel(file_path, sheet_name='input data')
    print("\nDataset loaded successfully!")
except FileNotFoundError:
    print("\nError: Dataset file not found.")
    exit()

# Data cleaning
df_clean = df.dropna(subset=['Year', 'Triplet_deliveries'])
df_clean['Year'] = df_clean['Year'].astype(int)

# Filter to analysis period
start_year = 1970
end_year = 2020
df_filtered = df_clean[(df_clean['Year'] >= start_year) & 
                        (df_clean['Year'] <= end_year)].copy()
df_filtered = df_filtered.sort_values('Year').reset_index(drop=True)

# Extract data
years = df_filtered['Year'].values.astype(float)
triplets = df_filtered['Triplet_deliveries'].values.astype(float)

n = len(years)

print(f"\nAnalysis Period: {int(years[0])} - {int(years[-1])}")
print(f"Number of Observations: n = {n} years")

print("\n--- Dataset Preview ---")
print(f"{'Year':>8}  {'Triplets':>12}")
print("-" * 25)
for i in range(min(10, n)):
    print(f"{int(years[i]):>8}  {triplets[i]:>12.0f}")
print("...")

# =============================================================================
# STEP 2: EXPLORATORY CORRELATION ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: EXPLORATORY CORRELATION ANALYSIS")
print("=" * 70)

print("\n--- Year-Triplets Correlation ---")

# Calculate Pearson correlation
r_pearson, p_pearson = stats.pearsonr(years, triplets)

# Calculate Spearman correlation
r_spearman, p_spearman = stats.spearmanr(years, triplets)

print(f"\n  Pearson Correlation:")
print(f"    r = {r_pearson:.6f}")
print(f"    p-value = {p_pearson:.6e}")

print(f"\n  Spearman Correlation:")
print(f"    ρ = {r_spearman:.6f}")
print(f"    p-value = {p_spearman:.6e}")

print(f"\n--- Interpretation ---")
strength = 'strong' if abs(r_pearson) > 0.7 else 'moderate' if abs(r_pearson) > 0.4 else 'weak'
direction = 'positive' if r_pearson > 0 else 'negative'
print(f"    There is a {strength} {direction} correlation between year and triplet births")

if abs(r_pearson - r_spearman) > 0.1:
    print("    Note: Different Pearson and Spearman values suggest non-linear relationship")

# =============================================================================
# STEP 3: SIMPLE LINEAR REGRESSION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: SIMPLE LINEAR REGRESSION")
print("=" * 70)

print("\n--- Model: Triplets = β₀ + β₁ × Year + ε ---")

# Fit linear regression using scipy
slope, intercept, r_value, p_value, std_err = stats.linregress(years, triplets)

# Calculate predictions
y_pred_linear = intercept + slope * years

# Calculate R² and other metrics
SS_tot = np.sum((triplets - np.mean(triplets)) ** 2)
SS_res_linear = np.sum((triplets - y_pred_linear) ** 2)
R2_linear = 1 - (SS_res_linear / SS_tot)

# Adjusted R²
R2_adj_linear = 1 - (1 - R2_linear) * (n - 1) / (n - 2)

# Root Mean Squared Error
RMSE_linear = np.sqrt(SS_res_linear / n)

print(f"\n  Regression Coefficients:")
print(f"    Intercept (β̂₀) = {intercept:.4f}")
print(f"    Slope (β̂₁) = {slope:.6f}")
print(f"\n  Regression Equation:")
print(f"    Triplets = {intercept:.2f} + {slope:.4f} × Year")

print(f"\n  Model Fit Statistics:")
print(f"    R² = {R2_linear:.6f}")
print(f"    Adjusted R² = {R2_adj_linear:.6f}")
print(f"    RMSE = {RMSE_linear:.4f}")
print(f"    Standard Error of Slope = {std_err:.6f}")

print(f"\n  Hypothesis Test (H₀: β₁ = 0):")
print(f"    t-statistic = {slope/std_err:.4f}")
print(f"    p-value = {p_value:.6e}")

# Predictions
print(f"\n--- Predictions ---")
future_years = [2025, 2030, 2035]
print(f"\n  {'Year':>8}  {'Predicted Triplets':>20}")
print("  " + "-" * 35)
for year in future_years:
    prediction = intercept + slope * year
    print(f"  {year:>8}  {prediction:>20.0f}")

# =============================================================================
# STEP 4: POLYNOMIAL REGRESSION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: POLYNOMIAL REGRESSION")
print("=" * 70)

print("\n--- Why Polynomial Regression? ---")
print("\n  Linear models may not capture complex trends.")
print("  Polynomial regression extends the linear model:")
print("    y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₚxᵖ")

# Normalize years for numerical stability
years_centered = years - np.mean(years)

# Fit polynomials of different degrees
results = {}
for degree in [1, 2, 3, 4]:
    coeffs = np.polyfit(years_centered, triplets, degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(years_centered)
    
    SS_res = np.sum((triplets - y_pred) ** 2)
    R2 = 1 - (SS_res / SS_tot)
    R2_adj = 1 - (1 - R2) * (n - 1) / (n - degree - 1)
    RMSE = np.sqrt(SS_res / n)
    
    # Calculate AIC and BIC
    k = degree + 1  # number of parameters
    AIC = n * np.log(SS_res / n) + 2 * k
    BIC = n * np.log(SS_res / n) + k * np.log(n)
    
    results[degree] = {
        'coeffs': coeffs,
        'poly': poly,
        'y_pred': y_pred,
        'R2': R2,
        'R2_adj': R2_adj,
        'RMSE': RMSE,
        'AIC': AIC,
        'BIC': BIC
    }

print("\n--- Model Comparison ---")
print("  " + "-" * 70)
print(f"  {'Degree':>8}  {'R²':>10}  {'Adj R²':>10}  {'RMSE':>10}  {'AIC':>12}  {'BIC':>12}")
print("  " + "-" * 70)
for degree, res in results.items():
    print(f"  {degree:>8}  {res['R2']:>10.6f}  {res['R2_adj']:>10.6f}  "
          f"{res['RMSE']:>10.2f}  {res['AIC']:>12.2f}  {res['BIC']:>12.2f}")

# Find best model
best_by_r2 = max(results.keys(), key=lambda d: results[d]['R2_adj'])
best_by_aic = min(results.keys(), key=lambda d: results[d]['AIC'])
best_by_bic = min(results.keys(), key=lambda d: results[d]['BIC'])

print(f"\n--- Best Model Selection ---")
print(f"    Best by Adjusted R²: Degree {best_by_r2}")
print(f"    Best by AIC: Degree {best_by_aic}")
print(f"    Best by BIC: Degree {best_by_bic}")

# =============================================================================
# STEP 5: QUADRATIC REGRESSION DETAILS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: QUADRATIC REGRESSION ANALYSIS")
print("=" * 70)

print("\n--- Model: Triplets = β₀ + β₁(Year - μ) + β₂(Year - μ)² ---")

quad_coeffs = results[2]['coeffs']
print(f"\n  Coefficients (with centered years):")
print(f"    β₂ (quadratic) = {quad_coeffs[0]:.6f}")
print(f"    β₁ (linear) = {quad_coeffs[1]:.6f}")
print(f"    β₀ (intercept) = {quad_coeffs[2]:.6f}")

# Find vertex (maximum or minimum point)
# For y = ax² + bx + c, vertex is at x = -b/(2a)
a, b, c = quad_coeffs[0], quad_coeffs[1], quad_coeffs[2]
vertex_x = -b / (2 * a)
vertex_year = vertex_x + np.mean(years)
vertex_y = a * vertex_x**2 + b * vertex_x + c

print(f"\n--- Critical Point (Vertex) ---")
print(f"    Vertex x (centered) = {vertex_x:.2f}")
print(f"    Vertex Year = {vertex_year:.1f}")
print(f"    Vertex y (predicted triplets) = {vertex_y:.0f}")

if a < 0:
    print(f"\n    Since β₂ < 0, the parabola opens downward")
    print(f"    Maximum triplet births occurred around {vertex_year:.0f}")
else:
    print(f"\n    Since β₂ > 0, the parabola opens upward")
    print(f"    Minimum triplet births occurred around {vertex_year:.0f}")

print(f"\n--- Model Quality ---")
print(f"    R² = {results[2]['R2']:.6f}")
print(f"    Adjusted R² = {results[2]['R2_adj']:.6f}")
print(f"    RMSE = {results[2]['RMSE']:.2f}")

# =============================================================================
# STEP 6: RESIDUAL DIAGNOSTICS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: RESIDUAL DIAGNOSTICS")
print("=" * 70)

# Use quadratic model residuals
residuals_quad = triplets - results[2]['y_pred']

print("\n--- Residual Summary ---")
print(f"    Mean: {np.mean(residuals_quad):.6f} (should be ≈ 0)")
print(f"    Std Dev: {np.std(residuals_quad):.4f}")
print(f"    Min: {np.min(residuals_quad):.4f}")
print(f"    Max: {np.max(residuals_quad):.4f}")

# Normality test
stat_shapiro, p_shapiro = stats.shapiro(residuals_quad)
print(f"\n--- Normality Test (Shapiro-Wilk) ---")
print(f"    W-statistic = {stat_shapiro:.6f}")
print(f"    p-value = {p_shapiro:.6f}")
if p_shapiro > 0.05:
    print("    Conclusion: Residuals appear normally distributed (p > 0.05)")
else:
    print("    Conclusion: Residuals may not be normally distributed (p ≤ 0.05)")

# Autocorrelation (Durbin-Watson)
def durbin_watson(residuals):
    """Calculate Durbin-Watson statistic."""
    diff = np.diff(residuals)
    return np.sum(diff**2) / np.sum(residuals**2)

dw_stat = durbin_watson(residuals_quad)
print(f"\n--- Autocorrelation Test (Durbin-Watson) ---")
print(f"    DW statistic = {dw_stat:.4f}")
print("    Interpretation:")
print("      DW ≈ 2: No autocorrelation")
print("      DW < 2: Positive autocorrelation")
print("      DW > 2: Negative autocorrelation")

if 1.5 < dw_stat < 2.5:
    print(f"    Conclusion: No significant autocorrelation detected")
else:
    print(f"    Conclusion: Potential autocorrelation in residuals")

# =============================================================================
# STEP 7: CONFIDENCE AND PREDICTION INTERVALS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: CONFIDENCE AND PREDICTION INTERVALS")
print("=" * 70)

print("\n--- Understanding Intervals ---")
print("\n  Confidence Interval: Range for the mean response")
print("  Prediction Interval: Range for an individual observation")
print("  Prediction intervals are always wider than confidence intervals")

# For linear regression
MSE = SS_res_linear / (n - 2)
se_slope = np.sqrt(MSE / np.sum((years - np.mean(years))**2))

# Calculate intervals for specific years
prediction_years = np.array([2000, 2010, 2020])

print("\n--- Linear Model Intervals ---")
print("  " + "-" * 70)
print(f"  {'Year':>6}  {'Prediction':>12}  {'95% CI':>20}  {'95% PI':>20}")
print("  " + "-" * 70)

for year in prediction_years:
    pred = intercept + slope * year
    
    # Standard error of prediction for mean
    se_fit = np.sqrt(MSE * (1/n + (year - np.mean(years))**2 / np.sum((years - np.mean(years))**2)))
    
    # Standard error for individual prediction
    se_pred = np.sqrt(MSE * (1 + 1/n + (year - np.mean(years))**2 / np.sum((years - np.mean(years))**2)))
    
    t_crit = stats.t.ppf(0.975, n - 2)
    
    ci_lower = pred - t_crit * se_fit
    ci_upper = pred + t_crit * se_fit
    pi_lower = pred - t_crit * se_pred
    pi_upper = pred + t_crit * se_pred
    
    print(f"  {int(year):>6}  {pred:>12.0f}  [{ci_lower:>8.0f}, {ci_upper:>8.0f}]  "
          f"[{pi_lower:>8.0f}, {pi_upper:>8.0f}]")

# =============================================================================
# STEP 8: MODEL SELECTION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: MODEL SELECTION CRITERIA")
print("=" * 70)

print("\n--- Information Criteria ---")
print("\n  AIC (Akaike Information Criterion):")
print("    AIC = n × ln(SSE/n) + 2k")
print("    Balances fit quality with model complexity")
print("    Lower is better")

print("\n  BIC (Bayesian Information Criterion):")
print("    BIC = n × ln(SSE/n) + k × ln(n)")
print("    Stronger penalty for model complexity than AIC")
print("    Lower is better")

print("\n--- Comparison Summary ---")
print("  " + "-" * 45)
print(f"  {'Criterion':>15}  {'Best Degree':>15}  {'Value':>12}")
print("  " + "-" * 45)
print(f"  {'Adjusted R²':>15}  {best_by_r2:>15}  {results[best_by_r2]['R2_adj']:>12.6f}")
print(f"  {'AIC':>15}  {best_by_aic:>15}  {results[best_by_aic]['AIC']:>12.2f}")
print(f"  {'BIC':>15}  {best_by_bic:>15}  {results[best_by_bic]['BIC']:>12.2f}")

# Final recommendation
print("\n--- Recommendation ---")
# Use majority voting or BIC (more conservative)
recommended_degree = best_by_bic
print(f"    Based on BIC (most conservative), recommend: Degree {recommended_degree}")
print(f"    This model captures the main trend while avoiding overfitting")

# =============================================================================
# STEP 9: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Scatter plot with regression lines
ax1 = axes[0, 0]
ax1.scatter(years, triplets, color='steelblue', s=50, alpha=0.7, 
            edgecolors='white', label='Observed', zorder=5)

# Plot linear and quadratic fits
colors = {'Linear': 'red', 'Quadratic': 'green', 'Cubic': 'orange'}
years_smooth = np.linspace(min(years), max(years), 200)
years_smooth_centered = years_smooth - np.mean(years)

ax1.plot(years_smooth, intercept + slope * years_smooth, 'r--', 
         linewidth=2, label=f'Linear (R²={R2_linear:.3f})')
ax1.plot(years_smooth, results[2]['poly'](years_smooth_centered), 'g-', 
         linewidth=2, label=f'Quadratic (R²={results[2]["R2"]:.3f})')

ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Triplet Deliveries', fontsize=11)
ax1.set_title('Regression Models Comparison\nLinear vs Quadratic', 
              fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Residual comparison
ax2 = axes[0, 1]
residuals_linear = triplets - y_pred_linear

ax2.scatter(years, residuals_linear, color='red', s=40, alpha=0.6, 
            label='Linear Residuals')
ax2.scatter(years, residuals_quad, color='green', s=40, alpha=0.6, 
            label='Quadratic Residuals')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Residual (y - ŷ)', fontsize=11)
ax2.set_title('Residual Analysis\nLinear vs Quadratic Models', 
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: R² comparison by polynomial degree
ax3 = axes[1, 0]
degrees = list(results.keys())
r2_values = [results[d]['R2'] for d in degrees]
r2_adj_values = [results[d]['R2_adj'] for d in degrees]

x_pos = np.arange(len(degrees))
width = 0.35

bars1 = ax3.bar(x_pos - width/2, r2_values, width, label='R²', color='steelblue')
bars2 = ax3.bar(x_pos + width/2, r2_adj_values, width, label='Adjusted R²', color='orange')

ax3.set_xlabel('Polynomial Degree', fontsize=11)
ax3.set_ylabel('R² Value', fontsize=11)
ax3.set_title('Model Fit by Polynomial Degree\nR² vs Adjusted R²', 
              fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(degrees)
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')
ax3.set_ylim(0, 1)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 4: Q-Q plot of quadratic residuals
ax4 = axes[1, 1]
stats.probplot(residuals_quad, dist="norm", plot=ax4)
ax4.set_title('Q-Q Plot of Quadratic Model Residuals\n(Check for Normality)', 
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/correlation_regression_triplet_births.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: correlation_regression_triplet_births.png")

# =============================================================================
# STEP 10: SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: SUMMARY")
print("=" * 70)

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│               CORRELATION AND REGRESSION SUMMARY                     │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  CORRELATION                                                         │")
print(f"│    Pearson r:                           {r_pearson:>10.6f}                │")
print(f"│    Spearman ρ:                          {r_spearman:>10.6f}                │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  LINEAR REGRESSION                                                   │")
print(f"│    Equation: Triplets = {intercept:.1f} + {slope:.3f} × Year                  │")
print(f"│    R²:                                  {R2_linear:>10.6f}                │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  QUADRATIC REGRESSION                                                │")
print(f"│    Peak Year (vertex):                  {vertex_year:>10.0f}                    │")
print(f"│    Peak Triplets:                       {vertex_y:>10.0f}                    │")
print(f"│    R²:                                  {results[2]['R2']:>10.6f}                │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  MODEL SELECTION                                                     │")
print(f"│    Best Model (BIC):                    Degree {best_by_bic:>4}                    │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\n--- Dataset Reference ---")
print("  Human Multiple Births Database (2024) FRA_InputData_25.11.2024.xlsx.")
print("  Available at: https://www.twinbirths.org/en/data-metadata/")
print("  (Accessed: 25 November 2024).")

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
