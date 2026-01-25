"""
================================================================================
HYPOTHESIS TESTING - TRIPLET BIRTHS DATASET
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
This script applies hypothesis testing to the triplet births dataset. We test
hypotheses about temporal trends, structural changes, and demographic patterns
in multiple birth data.

MATHEMATICAL CONCEPTS:
- Testing for trends in time series
- Structural break tests
- Comparing time periods
- Bootstrap hypothesis testing
- Multiple testing corrections

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
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# Define script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# HEADER INFORMATION
# =============================================================================

print("=" * 70)
print("HYPOTHESIS TESTING")
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

file_path = os.path.join(script_dir, '../datasets/FRA_InputData_25.11.2024.xlsx')

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

print("\n--- Descriptive Statistics ---")
print(f"    Mean Triplets: {np.mean(triplets):.2f}")
print(f"    Std Dev: {np.std(triplets, ddof=1):.2f}")
print(f"    Min: {np.min(triplets):.0f} (Year {int(years[np.argmin(triplets)])})")
print(f"    Max: {np.max(triplets):.0f} (Year {int(years[np.argmax(triplets)])})")

# =============================================================================
# STEP 2: TESTING FOR TREND
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: TESTING FOR TREND")
print("=" * 70)

print("\n--- Hypothesis: Is There a Linear Trend in Triplet Births? ---")
print("\n  H₀: β₁ = 0 (no linear trend)")
print("  H₁: β₁ ≠ 0 (linear trend exists)")

# Perform linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(years, triplets)

# Calculate t-statistic
t_stat = slope / std_err

print(f"\n--- Regression Results ---")
print(f"  Slope (β̂₁) = {slope:.6f}")
print(f"  Standard Error = {std_err:.6f}")
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value = {p_value:.6e}")

# Decision
alpha = 0.05
print(f"\n--- Decision (α = {alpha}) ---")
if p_value < alpha:
    print(f"  p-value ({p_value:.6e}) < α ({alpha})")
    print("  Reject H₀: There is a significant linear trend in triplet births")
    if slope > 0:
        print(f"  Direction: Increasing trend ({slope:.2f} more triplets per year)")
    else:
        print(f"  Direction: Decreasing trend ({slope:.2f} fewer triplets per year)")
else:
    print(f"  p-value ({p_value:.6e}) ≥ α ({alpha})")
    print("  Fail to reject H₀: No significant linear trend detected")

# =============================================================================
# STEP 3: MANN-KENDALL TREND TEST
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: MANN-KENDALL TREND TEST")
print("=" * 70)

print("\n--- Non-Parametric Trend Test ---")
print("\n  The Mann-Kendall test is robust to non-normal data and outliers")
print("  It tests for monotonic trends without assuming linearity")

def mann_kendall_test(data):
    """
    Perform Mann-Kendall trend test.
    
    Returns:
        S: Mann-Kendall statistic
        p: two-sided p-value
        trend: 'increasing', 'decreasing', or 'no trend'
    """
    n = len(data)
    s = 0
    
    # Calculate S statistic
    for i in range(n - 1):
        for j in range(i + 1, n):
            s += np.sign(data[j] - data[i])
    
    # Calculate variance
    # Account for ties
    unique, counts = np.unique(data, return_counts=True)
    tie_correction = np.sum(counts * (counts - 1) * (2 * counts + 5))
    
    var_s = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18
    
    # Calculate Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    # Calculate p-value (two-tailed)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Determine trend
    if p < 0.05:
        if s > 0:
            trend = 'increasing'
        else:
            trend = 'decreasing'
    else:
        trend = 'no trend'
    
    return s, z, p, trend

S, Z, p_mk, trend = mann_kendall_test(triplets)

print(f"\n--- Mann-Kendall Test Results ---")
print(f"  S statistic = {S:.0f}")
print(f"  Z statistic = {Z:.4f}")
print(f"  p-value = {p_mk:.6f}")
print(f"  Detected Trend: {trend}")

print(f"\n--- Interpretation ---")
if trend == 'no trend':
    print("  No significant monotonic trend detected in the data")
else:
    print(f"  Significant {trend} monotonic trend detected (p < 0.05)")

# =============================================================================
# STEP 4: COMPARING TIME PERIODS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: COMPARING TIME PERIODS")
print("=" * 70)

print("\n--- Testing: Did Triplet Births Change Between Decades? ---")

# Split data into decades
decades = {
    '1970s': (1970, 1979),
    '1980s': (1980, 1989),
    '1990s': (1990, 1999),
    '2000s': (2000, 2009),
    '2010s': (2010, 2019)
}

decade_data = {}
for decade_name, (start, end) in decades.items():
    mask = (years >= start) & (years <= end)
    if np.sum(mask) > 0:
        decade_data[decade_name] = triplets[mask]

print("\n--- Decade Summary Statistics ---")
print(f"  {'Decade':>8}  {'n':>4}  {'Mean':>10}  {'Std Dev':>10}  {'Min':>8}  {'Max':>8}")
print("  " + "-" * 60)
for decade, data in decade_data.items():
    print(f"  {decade:>8}  {len(data):>4}  {np.mean(data):>10.2f}  "
          f"{np.std(data, ddof=1):>10.2f}  {np.min(data):>8.0f}  {np.max(data):>8.0f}")

# One-way ANOVA
print("\n" + "-" * 50)
print("4.1 ONE-WAY ANOVA")
print("-" * 50)

print("\n  H₀: μ₁ = μ₂ = μ₃ = μ₄ = μ₅ (all decade means are equal)")
print("  H₁: At least one decade mean differs")

decade_arrays = list(decade_data.values())
F_stat, p_anova = stats.f_oneway(*decade_arrays)

print(f"\n  F-statistic = {F_stat:.4f}")
print(f"  p-value = {p_anova:.6e}")

if p_anova < 0.05:
    print("\n  Decision: Reject H₀")
    print("  At least one decade has significantly different mean triplet births")
else:
    print("\n  Decision: Fail to reject H₀")
    print("  No significant difference in means across decades")

# Post-hoc pairwise comparisons (if ANOVA significant)
if p_anova < 0.05:
    print("\n--- Post-hoc Pairwise Comparisons (Bonferroni corrected) ---")
    
    decade_names = list(decade_data.keys())
    n_comparisons = len(decade_names) * (len(decade_names) - 1) // 2
    alpha_bonf = 0.05 / n_comparisons
    
    print(f"\n  Number of comparisons: {n_comparisons}")
    print(f"  Bonferroni-corrected α: {alpha_bonf:.4f}")
    
    print(f"\n  {'Comparison':>20}  {'t-stat':>10}  {'p-value':>12}  {'Significant':>12}")
    print("  " + "-" * 60)
    
    for i in range(len(decade_names)):
        for j in range(i + 1, len(decade_names)):
            t_stat_pair, p_pair = stats.ttest_ind(
                decade_data[decade_names[i]], 
                decade_data[decade_names[j]]
            )
            sig = "Yes" if p_pair < alpha_bonf else "No"
            comparison = f"{decade_names[i]} vs {decade_names[j]}"
            print(f"  {comparison:>20}  {t_stat_pair:>10.4f}  {p_pair:>12.6f}  {sig:>12}")

# =============================================================================
# STEP 5: STRUCTURAL BREAK TEST
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: STRUCTURAL BREAK TEST")
print("=" * 70)

print("\n--- Testing for Structural Change in the Time Series ---")
print("\n  A structural break indicates a significant shift in the data pattern")
print("  We test if the regression relationship changed at some point")

def chow_test(y, x, break_point):
    """
    Perform Chow test for structural break.
    
    Parameters:
        y: dependent variable
        x: independent variable
        break_point: index where break is hypothesized
    
    Returns:
        F-statistic and p-value
    """
    n = len(y)
    k = 2  # number of parameters (intercept + slope)
    
    # Full model
    X_full = np.column_stack([np.ones(n), x])
    beta_full = np.linalg.lstsq(X_full, y, rcond=None)[0]
    y_pred_full = X_full @ beta_full
    SSR_full = np.sum((y - y_pred_full) ** 2)
    
    # Split models
    y1, x1 = y[:break_point], x[:break_point]
    y2, x2 = y[break_point:], x[break_point:]
    
    X1 = np.column_stack([np.ones(len(y1)), x1])
    X2 = np.column_stack([np.ones(len(y2)), x2])
    
    beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
    beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
    
    y_pred1 = X1 @ beta1
    y_pred2 = X2 @ beta2
    
    SSR1 = np.sum((y1 - y_pred1) ** 2)
    SSR2 = np.sum((y2 - y_pred2) ** 2)
    SSR_split = SSR1 + SSR2
    
    # F-statistic
    F = ((SSR_full - SSR_split) / k) / (SSR_split / (n - 2 * k))
    p_value = 1 - stats.f.cdf(F, k, n - 2 * k)
    
    return F, p_value

# Test for structural break around 1995 (peak of triplet births)
peak_year = years[np.argmax(triplets)]
break_idx = np.argmin(np.abs(years - peak_year))

print(f"\n  Testing for break at year {int(peak_year)} (peak triplet year)")
print(f"  H₀: No structural break at year {int(peak_year)}")
print(f"  H₁: Structural break exists at year {int(peak_year)}")

F_chow, p_chow = chow_test(triplets, years, break_idx)

print(f"\n--- Chow Test Results ---")
print(f"  F-statistic = {F_chow:.4f}")
print(f"  p-value = {p_chow:.6f}")

if p_chow < 0.05:
    print(f"\n  Decision: Reject H₀")
    print(f"  Evidence of structural break around {int(peak_year)}")
else:
    print(f"\n  Decision: Fail to reject H₀")
    print(f"  No significant structural break detected at {int(peak_year)}")

# =============================================================================
# STEP 6: NORMALITY TESTING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: NORMALITY TESTING")
print("=" * 70)

print("\n--- Testing if Triplet Data Follows Normal Distribution ---")
print("\n  H₀: Data is normally distributed")
print("  H₁: Data is not normally distributed")

# Multiple normality tests
print("\n" + "-" * 50)
print("6.1 SHAPIRO-WILK TEST")
print("-" * 50)

stat_shapiro, p_shapiro = stats.shapiro(triplets)
print(f"  W-statistic = {stat_shapiro:.6f}")
print(f"  p-value = {p_shapiro:.6f}")

print("\n" + "-" * 50)
print("6.2 D'AGOSTINO-PEARSON TEST")
print("-" * 50)

stat_dagostino, p_dagostino = stats.normaltest(triplets)
print(f"  K²-statistic = {stat_dagostino:.6f}")
print(f"  p-value = {p_dagostino:.6f}")

print("\n" + "-" * 50)
print("6.3 ANDERSON-DARLING TEST")
print("-" * 50)

result_anderson = stats.anderson(triplets, dist='norm')
print(f"  Statistic = {result_anderson.statistic:.6f}")
print("  Critical Values:")
for i, (cv, sl) in enumerate(zip(result_anderson.critical_values, 
                                   result_anderson.significance_level)):
    is_normal = result_anderson.statistic < cv
    print(f"    {sl:>5.1f}%: {cv:.4f} {'✓' if is_normal else '✗'}")

print("\n--- Summary ---")
if p_shapiro > 0.05 and p_dagostino > 0.05:
    print("  Conclusion: Data appears normally distributed")
else:
    print("  Conclusion: Data may not be normally distributed")
    print("  Consider using non-parametric tests")

# =============================================================================
# STEP 7: BOOTSTRAP HYPOTHESIS TEST
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: BOOTSTRAP HYPOTHESIS TEST")
print("=" * 70)

print("\n--- Bootstrap Method ---")
print("\n  Bootstrap resampling provides distribution-free inference")
print("  by repeatedly sampling with replacement from the data")

# Compare first and last decade using bootstrap
first_decade = triplets[years < 1980]
last_decade = triplets[years >= 2010]

observed_diff = np.mean(last_decade) - np.mean(first_decade)

print(f"\n  Comparing 1970s vs 2010s:")
print(f"  Mean 1970s: {np.mean(first_decade):.2f}")
print(f"  Mean 2010s: {np.mean(last_decade):.2f}")
print(f"  Observed Difference: {observed_diff:.2f}")

# Bootstrap
np.random.seed(42)
n_bootstrap = 10000
bootstrap_diffs = []

# Combine data under null hypothesis
combined = np.concatenate([first_decade, last_decade])
n1, n2 = len(first_decade), len(last_decade)

for _ in range(n_bootstrap):
    # Shuffle and split
    np.random.shuffle(combined)
    group1 = combined[:n1]
    group2 = combined[n1:]
    bootstrap_diffs.append(np.mean(group2) - np.mean(group1))

bootstrap_diffs = np.array(bootstrap_diffs)

# Calculate p-value (two-tailed)
p_bootstrap = 2 * min(
    np.mean(bootstrap_diffs >= observed_diff),
    np.mean(bootstrap_diffs <= observed_diff)
)

print(f"\n--- Bootstrap Results ({n_bootstrap:,} iterations) ---")
print(f"  Bootstrap Mean Difference: {np.mean(bootstrap_diffs):.4f}")
print(f"  Bootstrap Std Dev: {np.std(bootstrap_diffs):.4f}")
print(f"  p-value = {p_bootstrap:.6f}")

# Bootstrap confidence interval
ci_lower = np.percentile(bootstrap_diffs, 2.5)
ci_upper = np.percentile(bootstrap_diffs, 97.5)
print(f"  95% CI for difference: [{ci_lower:.2f}, {ci_upper:.2f}]")

print(f"\n--- Decision ---")
if p_bootstrap < 0.05:
    print("  Reject H₀: Significant difference between 1970s and 2010s")
else:
    print("  Fail to reject H₀: No significant difference detected")

# =============================================================================
# STEP 8: TESTING FOR AUTOCORRELATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: TESTING FOR AUTOCORRELATION")
print("=" * 70)

print("\n--- Ljung-Box Test for Autocorrelation ---")
print("\n  H₀: No autocorrelation in the time series")
print("  H₁: Autocorrelation exists")

def ljung_box_test(data, lags=10):
    """Perform Ljung-Box test for autocorrelation."""
    n = len(data)
    
    # Calculate autocorrelations
    mean_data = np.mean(data)
    variance = np.var(data)
    
    acf = []
    for lag in range(1, lags + 1):
        cov = np.sum((data[lag:] - mean_data) * (data[:-lag] - mean_data)) / n
        acf.append(cov / variance)
    
    # Ljung-Box Q statistic
    Q = n * (n + 2) * np.sum([(r**2) / (n - k) for k, r in enumerate(acf, 1)])
    
    # p-value from chi-square distribution
    p_value = 1 - stats.chi2.cdf(Q, lags)
    
    return Q, p_value, acf

Q_stat, p_lb, acf_values = ljung_box_test(triplets, lags=10)

print(f"\n  Q-statistic = {Q_stat:.4f}")
print(f"  p-value = {p_lb:.6f}")

print("\n  Autocorrelation Coefficients:")
print("    Lag    ACF")
print("    " + "-" * 15)
for i, acf in enumerate(acf_values[:5], 1):
    print(f"    {i:>3}    {acf:>7.4f}")

if p_lb < 0.05:
    print("\n  Decision: Reject H₀")
    print("  Significant autocorrelation detected in the time series")
else:
    print("\n  Decision: Fail to reject H₀")
    print("  No significant autocorrelation detected")

# =============================================================================
# STEP 9: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Time series with trend line and confidence band
ax1 = axes[0, 0]
ax1.scatter(years, triplets, color='steelblue', s=50, alpha=0.7, 
            edgecolors='white', label='Observed', zorder=5)

# Trend line
y_pred = intercept + slope * years
ax1.plot(years, y_pred, 'r-', linewidth=2, label=f'Trend (p={p_value:.2e})')

# Mark structural break
ax1.axvline(x=peak_year, color='green', linestyle='--', linewidth=2,
           alpha=0.7, label=f'Peak Year: {int(peak_year)}')

ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Triplet Deliveries', fontsize=11)
ax1.set_title('Trend Analysis with Structural Break Test', 
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Decade comparison boxplot
ax2 = axes[0, 1]
decade_labels = list(decade_data.keys())
decade_arrays = list(decade_data.values())

bp = ax2.boxplot(decade_arrays, labels=decade_labels, patch_artist=True)
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(decade_labels)))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Add significance bars if ANOVA was significant
if p_anova < 0.05:
    ax2.text(0.5, 0.98, f'ANOVA p = {p_anova:.4f} *', transform=ax2.transAxes,
            ha='center', va='top', fontsize=10, color='red')

ax2.set_xlabel('Decade', fontsize=11)
ax2.set_ylabel('Triplet Deliveries', fontsize=11)
ax2.set_title('Comparison Across Decades\n(One-way ANOVA)', 
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Bootstrap distribution
ax3 = axes[1, 0]
ax3.hist(bootstrap_diffs, bins=50, density=True, alpha=0.7, 
         color='steelblue', edgecolor='white', label='Bootstrap Distribution')
ax3.axvline(x=observed_diff, color='red', linestyle='--', linewidth=2,
           label=f'Observed Diff: {observed_diff:.1f}')
ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)

# Add shaded rejection regions
ax3.axvline(x=ci_lower, color='orange', linestyle=':', linewidth=1.5)
ax3.axvline(x=ci_upper, color='orange', linestyle=':', linewidth=1.5,
           label=f'95% CI: [{ci_lower:.1f}, {ci_upper:.1f}]')

ax3.set_xlabel('Difference in Means (2010s - 1970s)', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title(f'Bootstrap Hypothesis Test\n(p = {p_bootstrap:.4f})', 
              fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)

# Plot 4: Q-Q plot for normality
ax4 = axes[1, 1]
stats.probplot(triplets, dist="norm", plot=ax4)
ax4.set_title(f'Q-Q Plot for Normality\nShapiro-Wilk p = {p_shapiro:.4f}', 
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
# Ensure output directory exists
output_dir = os.path.join(script_dir, '../outputs/figures')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(script_dir, '../outputs/figures/hypothesis_testing_triplet_births.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: hypothesis_testing_triplet_births.png")

# =============================================================================
# STEP 10: SUMMARY OF ALL TESTS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: SUMMARY OF ALL HYPOTHESIS TESTS")
print("=" * 70)

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│                    HYPOTHESIS TESTING SUMMARY                        │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  TEST                          STATISTIC      P-VALUE    DECISION   │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│  Linear Trend (t-test)         t = {t_stat:>7.3f}    {p_value:>9.2e}  "
      f"{'Reject' if p_value < 0.05 else 'Fail':>8} │")
print(f"│  Mann-Kendall Trend            Z = {Z:>7.3f}    {p_mk:>9.6f}  "
      f"{'Reject' if p_mk < 0.05 else 'Fail':>8} │")
print(f"│  ANOVA (Decades)               F = {F_stat:>7.3f}    {p_anova:>9.2e}  "
      f"{'Reject' if p_anova < 0.05 else 'Fail':>8} │")
print(f"│  Chow Break Test               F = {F_chow:>7.3f}    {p_chow:>9.6f}  "
      f"{'Reject' if p_chow < 0.05 else 'Fail':>8} │")
print(f"│  Shapiro-Wilk Normality        W = {stat_shapiro:>7.4f}    {p_shapiro:>9.6f}  "
      f"{'Reject' if p_shapiro < 0.05 else 'Fail':>8} │")
print(f"│  Bootstrap (1970s vs 2010s)    Δ = {observed_diff:>7.1f}    {p_bootstrap:>9.6f}  "
      f"{'Reject' if p_bootstrap < 0.05 else 'Fail':>8} │")
print(f"│  Ljung-Box Autocorrelation     Q = {Q_stat:>7.2f}    {p_lb:>9.6f}  "
      f"{'Reject' if p_lb < 0.05 else 'Fail':>8} │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\n--- Key Findings ---")
if p_value < 0.05:
    print(f"  1. Significant linear trend detected (slope = {slope:.2f})")
if p_anova < 0.05:
    print("  2. Significant differences exist between decades")
if p_chow < 0.05:
    print(f"  3. Structural break detected around {int(peak_year)}")
if p_shapiro < 0.05:
    print("  4. Data may not be normally distributed")
if p_lb < 0.05:
    print("  5. Significant autocorrelation in time series")

print("\n--- Dataset Reference ---")
print("  Human Multiple Births Database (2024) FRA_InputData_25.11.2024.xlsx.")
print("  Available at: https://www.twinbirths.org/en/data-metadata/")
print("  (Accessed: 25 November 2024).")

print("\n--- Further Reading ---")
print("  Wasserman, L. (2004) All of Statistics. Springer.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF HYPOTHESIS TESTING ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
