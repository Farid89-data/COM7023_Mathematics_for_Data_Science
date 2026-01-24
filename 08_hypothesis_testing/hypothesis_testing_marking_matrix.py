"""
================================================================================
HYPOTHESIS TESTING - MARKING MATRIX DATASET
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
This script demonstrates hypothesis testing concepts using the marking matrix
dataset. We explore various statistical tests including t-tests, chi-square
tests, and non-parametric alternatives.

MATHEMATICAL CONCEPTS:
- Null and alternative hypotheses
- Test statistics (z, t, chi-square)
- P-values and significance levels
- Type I and Type II errors
- Confidence intervals
- Non-parametric tests

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
from scipy import stats
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
print("HYPOTHESIS TESTING")
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

# Define grade columns
grade_columns = ['90- 100', '80 - 90', '70 – 79', '60 - 69', 
                 '50 – 59', '40 – 49', '30 – 39', '20 - 29', '0 - 19']

grade_midpoints = {
    '90- 100': 95, '80 - 90': 85, '70 – 79': 74.5, '60 - 69': 64.5,
    '50 – 59': 54.5, '40 – 49': 44.5, '30 – 39': 34.5, '20 - 29': 24.5, '0 - 19': 9.5
}

# Create word count matrix
word_counts = []
for idx, row in df.iterrows():
    row_counts = []
    for grade_col in grade_columns:
        if grade_col in df.columns:
            text = str(row[grade_col])
            word_count = len(text.split())
            row_counts.append(word_count)
    word_counts.append(row_counts)

word_counts = np.array(word_counts)
n_los = word_counts.shape[0]

print(f"\nNumber of Learning Outcomes: {n_los}")
print(f"Number of Grade Bands: {len(grade_columns)}")
print(f"\nWord Counts Matrix Shape: {word_counts.shape}")

# =============================================================================
# STEP 2: HYPOTHESIS TESTING FRAMEWORK
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: HYPOTHESIS TESTING FRAMEWORK")
print("=" * 70)

print("\n--- The Scientific Method ---")
print("\n  1. State the null hypothesis (H₀) and alternative hypothesis (H₁)")
print("  2. Choose significance level (α)")
print("  3. Collect data and calculate test statistic")
print("  4. Compute p-value or compare to critical value")
print("  5. Make decision: reject or fail to reject H₀")

print("\n--- Key Definitions ---")
print("\n  Null Hypothesis (H₀): Statement of 'no effect' or 'no difference'")
print("  Alternative Hypothesis (H₁): Statement we want to provide evidence for")
print("  Significance Level (α): Probability of Type I error (usually 0.05)")
print("  P-value: Probability of observing data as extreme or more, given H₀ is true")

print("\n--- Decision Rule ---")
print("\n  If p-value < α: Reject H₀ (statistically significant)")
print("  If p-value ≥ α: Fail to reject H₀ (not statistically significant)")

print("\n--- Types of Errors ---")
print("\n  Type I Error (False Positive): Reject H₀ when H₀ is true")
print("    Probability = α")
print("\n  Type II Error (False Negative): Fail to reject H₀ when H₀ is false")
print("    Probability = β")
print("    Power = 1 - β (probability of correctly rejecting false H₀)")

# =============================================================================
# STEP 3: ONE-SAMPLE T-TEST
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: ONE-SAMPLE T-TEST")
print("=" * 70)

print("\n--- When to Use ---")
print("  - Testing if a sample mean differs from a known value")
print("  - Population standard deviation is unknown")
print("  - Data is approximately normally distributed")

print("\n--- Test Statistic ---")
print("\n           x̄ - μ₀")
print("    t = ─────────────")
print("          s / √n")
print("\n  Degrees of freedom: df = n - 1")

# Calculate mean word count for high grade bands (90-100, 80-90)
high_grade_words = word_counts[:, :2].flatten()
n = len(high_grade_words)
sample_mean = np.mean(high_grade_words)
sample_std = np.std(high_grade_words, ddof=1)
hypothesized_mean = 30  # Testing if average differs from 30 words

print(f"\n--- Example: Testing if High Grade Band Descriptions Average 30 Words ---")
print(f"\n  H₀: μ = {hypothesized_mean} (mean word count is 30)")
print(f"  H₁: μ ≠ {hypothesized_mean} (mean word count is not 30)")
print(f"  α = 0.05 (two-tailed test)")

print(f"\n--- Calculation ---")
print(f"  Sample size: n = {n}")
print(f"  Sample mean: x̄ = {sample_mean:.4f}")
print(f"  Sample std dev: s = {sample_std:.4f}")
print(f"  Hypothesized mean: μ₀ = {hypothesized_mean}")

t_statistic = (sample_mean - hypothesized_mean) / (sample_std / np.sqrt(n))
df = n - 1
p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))

print(f"\n  t = ({sample_mean:.4f} - {hypothesized_mean}) / ({sample_std:.4f} / √{n})")
print(f"  t = {sample_mean - hypothesized_mean:.4f} / {sample_std / np.sqrt(n):.4f}")
print(f"  t = {t_statistic:.4f}")

print(f"\n  Degrees of freedom: df = {df}")
print(f"  P-value (two-tailed) = {p_value:.6f}")

# Critical value
t_critical = stats.t.ppf(0.975, df)
print(f"\n  Critical value (α/2 = 0.025): t_crit = ±{t_critical:.4f}")

# Verification with scipy
t_scipy, p_scipy = stats.ttest_1samp(high_grade_words, hypothesized_mean)
print(f"\n--- Verification (scipy.stats.ttest_1samp) ---")
print(f"  t-statistic = {t_scipy:.4f}")
print(f"  p-value = {p_scipy:.6f}")

# Decision
alpha = 0.05
print(f"\n--- Decision (α = {alpha}) ---")
if p_value < alpha:
    print(f"  P-value ({p_value:.6f}) < α ({alpha})")
    print(f"  Reject H₀: Evidence suggests mean differs from {hypothesized_mean}")
else:
    print(f"  P-value ({p_value:.6f}) ≥ α ({alpha})")
    print(f"  Fail to reject H₀: Insufficient evidence mean differs from {hypothesized_mean}")

# =============================================================================
# STEP 4: TWO-SAMPLE T-TEST
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: TWO-SAMPLE T-TEST (INDEPENDENT SAMPLES)")
print("=" * 70)

print("\n--- When to Use ---")
print("  - Comparing means of two independent groups")
print("  - Testing if there's a significant difference between groups")

print("\n--- Test Statistic (Equal Variances) ---")
print("\n           x̄₁ - x̄₂")
print("    t = ──────────────────────")
print("        sp × √(1/n₁ + 1/n₂)")
print("\n  where sp is the pooled standard deviation")

# Compare high grade bands (90-100, 80-90) vs low grade bands (30-39, 20-29, 0-19)
high_grades = word_counts[:, :2].flatten()
low_grades = word_counts[:, -3:].flatten()

print(f"\n--- Example: Comparing High vs Low Grade Band Word Counts ---")
print(f"\n  H₀: μ_high = μ_low (no difference in mean word counts)")
print(f"  H₁: μ_high ≠ μ_low (difference exists)")
print(f"  α = 0.05")

print(f"\n--- Sample Statistics ---")
print(f"  High Grade Bands (90-100, 80-90):")
print(f"    n₁ = {len(high_grades)}")
print(f"    x̄₁ = {np.mean(high_grades):.4f}")
print(f"    s₁ = {np.std(high_grades, ddof=1):.4f}")

print(f"\n  Low Grade Bands (30-39, 20-29, 0-19):")
print(f"    n₂ = {len(low_grades)}")
print(f"    x̄₂ = {np.mean(low_grades):.4f}")
print(f"    s₂ = {np.std(low_grades, ddof=1):.4f}")

# Levene's test for equality of variances
stat_levene, p_levene = stats.levene(high_grades, low_grades)
print(f"\n--- Levene's Test for Equal Variances ---")
print(f"  F-statistic = {stat_levene:.4f}")
print(f"  p-value = {p_levene:.4f}")

equal_var = p_levene > 0.05
print(f"  Equal variances assumed: {equal_var}")

# Perform t-test
t_stat, p_value_2samp = stats.ttest_ind(high_grades, low_grades, equal_var=equal_var)

print(f"\n--- Two-Sample t-Test Results ---")
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value = {p_value_2samp:.6f}")

# Decision
print(f"\n--- Decision (α = 0.05) ---")
if p_value_2samp < 0.05:
    print(f"  P-value ({p_value_2samp:.6f}) < α (0.05)")
    print("  Reject H₀: Significant difference between high and low grade descriptions")
else:
    print(f"  P-value ({p_value_2samp:.6f}) ≥ α (0.05)")
    print("  Fail to reject H₀: No significant difference detected")

# Effect size (Cohen's d)
pooled_std = np.sqrt(((len(high_grades)-1)*np.var(high_grades, ddof=1) + 
                       (len(low_grades)-1)*np.var(low_grades, ddof=1)) / 
                      (len(high_grades) + len(low_grades) - 2))
cohens_d = (np.mean(high_grades) - np.mean(low_grades)) / pooled_std

print(f"\n--- Effect Size (Cohen's d) ---")
print(f"  d = {cohens_d:.4f}")
if abs(cohens_d) < 0.2:
    print("  Interpretation: Negligible effect")
elif abs(cohens_d) < 0.5:
    print("  Interpretation: Small effect")
elif abs(cohens_d) < 0.8:
    print("  Interpretation: Medium effect")
else:
    print("  Interpretation: Large effect")

# =============================================================================
# STEP 5: PAIRED T-TEST
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: PAIRED T-TEST")
print("=" * 70)

print("\n--- When to Use ---")
print("  - Comparing two measurements on the same subjects")
print("  - Before-after comparisons")
print("  - Matched pairs design")

print("\n--- Test Statistic ---")
print("\n           d̄")
print("    t = ─────────")
print("        sd / √n")
print("\n  where d̄ is the mean of differences")

# Compare adjacent grade bands for each LO
grade_95 = word_counts[:, 0]  # 90-100
grade_85 = word_counts[:, 1]  # 80-90

differences = grade_95 - grade_85
d_mean = np.mean(differences)
d_std = np.std(differences, ddof=1)
n_pairs = len(differences)

print(f"\n--- Example: Comparing Adjacent Grade Bands (90-100 vs 80-90) ---")
print(f"\n  H₀: μ_d = 0 (no difference between adjacent grade descriptions)")
print(f"  H₁: μ_d ≠ 0 (difference exists)")

print(f"\n--- Difference Statistics ---")
print(f"  Differences: {differences}")
print(f"  Mean difference (d̄) = {d_mean:.4f}")
print(f"  Std of differences (sd) = {d_std:.4f}")
print(f"  Number of pairs (n) = {n_pairs}")

t_paired = d_mean / (d_std / np.sqrt(n_pairs))
df_paired = n_pairs - 1
p_paired = 2 * (1 - stats.t.cdf(abs(t_paired), df_paired))

print(f"\n--- Calculation ---")
print(f"  t = {d_mean:.4f} / ({d_std:.4f} / √{n_pairs})")
print(f"  t = {t_paired:.4f}")
print(f"  df = {df_paired}")
print(f"  p-value = {p_paired:.6f}")

# Verification with scipy
t_scipy_paired, p_scipy_paired = stats.ttest_rel(grade_95, grade_85)
print(f"\n--- Verification (scipy.stats.ttest_rel) ---")
print(f"  t-statistic = {t_scipy_paired:.4f}")
print(f"  p-value = {p_scipy_paired:.6f}")

# =============================================================================
# STEP 6: CHI-SQUARE TEST
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: CHI-SQUARE TEST")
print("=" * 70)

print("\n--- Chi-Square Goodness of Fit Test ---")
print("\n  Tests if observed frequencies match expected frequencies")
print("\n  Test Statistic:")
print("              (Oᵢ - Eᵢ)²")
print("    χ² = Σ  ─────────────")
print("               Eᵢ")

# Create categorical data: classify word counts as Low, Medium, High
all_words = word_counts.flatten()
categories = pd.cut(all_words, bins=[0, 20, 40, 100], labels=['Low', 'Medium', 'High'])
observed_freq = categories.value_counts().sort_index()

print(f"\n--- Example: Testing if Word Count Categories are Equally Distributed ---")
print(f"\n  H₀: Word counts are equally distributed across categories")
print(f"  H₁: Word counts are not equally distributed")

print(f"\n--- Observed Frequencies ---")
for cat, freq in observed_freq.items():
    print(f"    {cat}: {freq}")

total = observed_freq.sum()
expected_freq = np.array([total / 3] * 3)  # Equal distribution

print(f"\n--- Expected Frequencies (under H₀) ---")
for cat, freq in zip(['Low', 'Medium', 'High'], expected_freq):
    print(f"    {cat}: {freq:.2f}")

# Calculate chi-square statistic
chi_square = np.sum((observed_freq.values - expected_freq) ** 2 / expected_freq)
df_chi = len(expected_freq) - 1
p_chi = 1 - stats.chi2.cdf(chi_square, df_chi)

print(f"\n--- Calculation ---")
print(f"  χ² = Σ (Oᵢ - Eᵢ)² / Eᵢ")
for i, (obs, exp) in enumerate(zip(observed_freq.values, expected_freq)):
    contribution = (obs - exp) ** 2 / exp
    print(f"    Category {i+1}: ({obs} - {exp:.2f})² / {exp:.2f} = {contribution:.4f}")
print(f"  χ² = {chi_square:.4f}")
print(f"  df = {df_chi}")
print(f"  p-value = {p_chi:.6f}")

# Verification with scipy
chi2_scipy, p_scipy_chi = stats.chisquare(observed_freq.values, expected_freq)
print(f"\n--- Verification (scipy.stats.chisquare) ---")
print(f"  χ² = {chi2_scipy:.4f}")
print(f"  p-value = {p_scipy_chi:.6f}")

# =============================================================================
# STEP 7: CONFIDENCE INTERVALS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: CONFIDENCE INTERVALS")
print("=" * 70)

print("\n--- Definition ---")
print("\n  A confidence interval provides a range of plausible values for a")
print("  population parameter. A 95% CI means: if we repeated the sampling")
print("  many times, 95% of the intervals would contain the true parameter.")

print("\n--- Formula for Mean (σ unknown) ---")
print("\n            s")
print("  x̄ ± t_α/2 × ───")
print("            √n")

# Calculate 95% CI for high grade word counts
sample_data = high_grade_words
n = len(sample_data)
x_bar = np.mean(sample_data)
s = np.std(sample_data, ddof=1)
se = s / np.sqrt(n)

confidence_level = 0.95
alpha = 1 - confidence_level
t_crit = stats.t.ppf(1 - alpha/2, n - 1)

ci_lower = x_bar - t_crit * se
ci_upper = x_bar + t_crit * se

print(f"\n--- 95% Confidence Interval for High Grade Word Counts ---")
print(f"\n  Sample Statistics:")
print(f"    n = {n}")
print(f"    x̄ = {x_bar:.4f}")
print(f"    s = {s:.4f}")
print(f"    SE = s/√n = {se:.4f}")

print(f"\n  Critical Value:")
print(f"    t_0.025,{n-1} = {t_crit:.4f}")

print(f"\n  Confidence Interval:")
print(f"    CI = {x_bar:.4f} ± {t_crit:.4f} × {se:.4f}")
print(f"    CI = {x_bar:.4f} ± {t_crit * se:.4f}")
print(f"    CI = [{ci_lower:.4f}, {ci_upper:.4f}]")

print(f"\n  Interpretation:")
print(f"    We are 95% confident that the true mean word count for high")
print(f"    grade bands is between {ci_lower:.2f} and {ci_upper:.2f}")

# Different confidence levels
print(f"\n--- Confidence Intervals at Different Levels ---")
print(f"  {'Level':>8}  {'t-critical':>12}  {'Margin':>10}  {'Interval':>25}")
print("  " + "-" * 60)
for level in [0.90, 0.95, 0.99]:
    t_c = stats.t.ppf(1 - (1-level)/2, n - 1)
    margin = t_c * se
    lower = x_bar - margin
    upper = x_bar + margin
    print(f"  {level*100:>7.0f}%  {t_c:>12.4f}  {margin:>10.4f}  [{lower:.4f}, {upper:.4f}]")

# =============================================================================
# STEP 8: NON-PARAMETRIC TESTS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: NON-PARAMETRIC TESTS")
print("=" * 70)

print("\n--- When to Use Non-Parametric Tests ---")
print("  - Data is not normally distributed")
print("  - Small sample sizes")
print("  - Ordinal or ranked data")
print("  - Outliers present")

print("\n" + "-" * 50)
print("8.1 MANN-WHITNEY U TEST")
print("-" * 50)

print("\n  Non-parametric alternative to independent samples t-test")
print("  Tests if one distribution is stochastically greater than the other")

stat_mw, p_mw = stats.mannwhitneyu(high_grades, low_grades, alternative='two-sided')

print(f"\n  Comparing High vs Low Grade Bands:")
print(f"    U-statistic = {stat_mw:.4f}")
print(f"    p-value = {p_mw:.6f}")

if p_mw < 0.05:
    print("    Result: Significant difference detected")
else:
    print("    Result: No significant difference detected")

print("\n" + "-" * 50)
print("8.2 WILCOXON SIGNED-RANK TEST")
print("-" * 50)

print("\n  Non-parametric alternative to paired t-test")
print("  Tests if the median difference is zero")

stat_wilcox, p_wilcox = stats.wilcoxon(grade_95, grade_85)

print(f"\n  Comparing Adjacent Grade Bands (90-100 vs 80-90):")
print(f"    W-statistic = {stat_wilcox:.4f}")
print(f"    p-value = {p_wilcox:.6f}")

if p_wilcox < 0.05:
    print("    Result: Significant difference in medians")
else:
    print("    Result: No significant difference in medians")

print("\n" + "-" * 50)
print("8.3 KRUSKAL-WALLIS TEST")
print("-" * 50)

print("\n  Non-parametric alternative to one-way ANOVA")
print("  Tests if multiple independent samples have the same distribution")

# Group word counts by grade category
low = word_counts[:, :3].flatten()    # High grades
mid = word_counts[:, 3:6].flatten()   # Medium grades
high = word_counts[:, 6:].flatten()   # Low grades

stat_kw, p_kw = stats.kruskal(low, mid, high)

print(f"\n  Comparing Three Grade Categories:")
print(f"    H-statistic = {stat_kw:.4f}")
print(f"    p-value = {p_kw:.6f}")

if p_kw < 0.05:
    print("    Result: At least one group differs significantly")
else:
    print("    Result: No significant differences among groups")

# =============================================================================
# STEP 9: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: One-sample t-test visualization
ax1 = axes[0, 0]
x_range = np.linspace(-4, 4, 1000)
y_t = stats.t.pdf(x_range, df)

ax1.plot(x_range, y_t, 'b-', linewidth=2, label=f't-distribution (df={df})')
ax1.fill_between(x_range[x_range <= -t_critical], y_t[x_range <= -t_critical], 
                alpha=0.3, color='red', label='Rejection Region (α/2)')
ax1.fill_between(x_range[x_range >= t_critical], y_t[x_range >= t_critical], 
                alpha=0.3, color='red')
ax1.axvline(x=t_statistic, color='green', linestyle='--', linewidth=2,
           label=f't-statistic = {t_statistic:.2f}')
ax1.axvline(x=t_critical, color='red', linestyle=':', linewidth=1.5)
ax1.axvline(x=-t_critical, color='red', linestyle=':', linewidth=1.5)

ax1.set_xlabel('t-value', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('One-Sample t-Test\nH₀: μ = 30', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Two-sample comparison
ax2 = axes[0, 1]
data_to_plot = [high_grades, low_grades]
bp = ax2.boxplot(data_to_plot, labels=['High Grades\n(90-100, 80-90)', 
                                        'Low Grades\n(30-39, 20-29, 0-19)'],
                 patch_artist=True)
colors = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax2.set_ylabel('Word Count', fontsize=11)
ax2.set_title(f'Two-Sample t-Test\np-value = {p_value_2samp:.4f}', 
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add significance indicator
if p_value_2samp < 0.05:
    max_y = max(max(high_grades), max(low_grades))
    ax2.plot([1, 1, 2, 2], [max_y + 2, max_y + 4, max_y + 4, max_y + 2], 'k-')
    ax2.text(1.5, max_y + 5, '*', ha='center', fontsize=20)

# Plot 3: Chi-square test visualization
ax3 = axes[1, 0]
x_pos = np.arange(3)
width = 0.35

bars1 = ax3.bar(x_pos - width/2, observed_freq.values, width, 
               label='Observed', color='steelblue')
bars2 = ax3.bar(x_pos + width/2, expected_freq, width, 
               label='Expected', color='orange', alpha=0.7)

ax3.set_xlabel('Category', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title(f'Chi-Square Goodness of Fit Test\nχ² = {chi_square:.2f}, p = {p_chi:.4f}', 
              fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['Low\n(0-20)', 'Medium\n(20-40)', 'High\n(40+)'])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Confidence interval visualization
ax4 = axes[1, 1]
levels = [0.90, 0.95, 0.99]
colors = ['lightgreen', 'skyblue', 'lightcoral']

for i, (level, color) in enumerate(zip(levels, colors)):
    t_c = stats.t.ppf(1 - (1-level)/2, n - 1)
    margin = t_c * se
    lower = x_bar - margin
    upper = x_bar + margin
    
    ax4.barh(i, upper - lower, left=lower, height=0.3, 
            color=color, alpha=0.7, edgecolor='black')
    ax4.plot([x_bar], [i], 'ko', markersize=8)
    ax4.text(upper + 0.5, i, f'{level*100:.0f}% CI: [{lower:.1f}, {upper:.1f}]', 
            va='center', fontsize=10)

ax4.axvline(x=x_bar, color='red', linestyle='--', linewidth=2,
           label=f'Sample Mean = {x_bar:.2f}')
ax4.set_yticks(range(len(levels)))
ax4.set_yticklabels([f'{l*100:.0f}%' for l in levels])
ax4.set_xlabel('Word Count', fontsize=11)
ax4.set_ylabel('Confidence Level', fontsize=11)
ax4.set_title('Confidence Intervals for Mean Word Count', 
              fontsize=12, fontweight='bold')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('../outputs/figures/hypothesis_testing_marking_matrix.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: hypothesis_testing_marking_matrix.png")

# =============================================================================
# STEP 10: SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: SUMMARY")
print("=" * 70)

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│                    HYPOTHESIS TESTING SUMMARY                        │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  ONE-SAMPLE T-TEST (H₀: μ = 30)                                      │")
print(f"│    t-statistic:                         {t_statistic:>10.4f}                    │")
print(f"│    p-value:                             {p_value:>10.6f}                │")
print(f"│    Decision:                      {'Reject H₀' if p_value < 0.05 else 'Fail to Reject':>15}              │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  TWO-SAMPLE T-TEST (High vs Low Grades)                              │")
print(f"│    t-statistic:                         {t_stat:>10.4f}                    │")
print(f"│    p-value:                             {p_value_2samp:>10.6f}                │")
print(f"│    Cohen's d:                           {cohens_d:>10.4f}                    │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  CHI-SQUARE TEST (Equal Distribution)                                │")
print(f"│    χ²-statistic:                        {chi_square:>10.4f}                    │")
print(f"│    p-value:                             {p_chi:>10.6f}                │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  95% CONFIDENCE INTERVAL                                             │")
print(f"│    Mean ± Margin:                   [{ci_lower:>6.2f}, {ci_upper:>6.2f}]              │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\n--- Dataset Reference ---")
print("  Arden University (2024) COM7023 Mathematics for Data Science")
print("  Marking Matrix. Arden University.")

print("\n--- Further Reading ---")
print("  Wasserman, L. (2004) All of Statistics. Springer.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF HYPOTHESIS TESTING ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
