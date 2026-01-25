"""
================================================================================
DESCRIPTIVE STATISTICS - TRIPLET BIRTHS DATASET
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
This script demonstrates descriptive statistics applied to real-world
demographic data on triplet births. The analysis covers all fundamental
measures of central tendency, dispersion, and distributional shape.

MATHEMATICAL CONCEPTS:
- Measures of central tendency: mean, median, mode
- Measures of dispersion: variance, standard deviation, range, IQR
- Shape measures: skewness, kurtosis
- Five-number summary

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

# Define script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# HEADER INFORMATION
# =============================================================================

print("=" * 70)
print("DESCRIPTIVE STATISTICS")
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
except FileNotFoundError:
    print("\nError: Dataset file not found.")
    exit()

# Data cleaning and filtering
df_clean = df.dropna(subset=['Year', 'Triplet_deliveries'])
df_clean['Year'] = df_clean['Year'].astype(int)

# Filter to analysis period
start_year = 1990
end_year = 2020
df_filtered = df_clean[(df_clean['Year'] >= start_year) & 
                        (df_clean['Year'] <= end_year)].copy()

# Extract the data for analysis
triplet_data = df_filtered['Triplet_deliveries'].values
years = df_filtered['Year'].values

print(f"Analysis period: {start_year} - {end_year}")
print(f"Number of observations: {len(triplet_data)}")

# =============================================================================
# STEP 2: MEASURES OF CENTRAL TENDENCY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: MEASURES OF CENTRAL TENDENCY")
print("=" * 70)

print("\n" + "-" * 50)
print("ARITHMETIC MEAN")
print("-" * 50)

print("\n  Formula: x̄ = (1/n) × Σxᵢ")
print("\n  The arithmetic mean is the sum of all observations")
print("  divided by the number of observations.")

# Step-by-step calculation
n = len(triplet_data)
sum_x = np.sum(triplet_data)
mean_x = sum_x / n

print(f"\n  Step-by-step calculation:")
print(f"  n = {n}")
print(f"  Σxᵢ = {sum_x:.0f}")
print(f"  x̄ = {sum_x:.0f} / {n} = {mean_x:.2f}")

print("\n" + "-" * 50)
print("MEDIAN")
print("-" * 50)

print("\n  The median is the middle value when data is ordered.")
print("  For even n: median = (x_{n/2} + x_{n/2+1}) / 2")
print("  For odd n: median = x_{(n+1)/2}")

# Sort the data
sorted_data = np.sort(triplet_data)
print(f"\n  Sorted data (first 10 values): {sorted_data[:10]}")
print(f"  Sorted data (last 10 values): {sorted_data[-10:]}")

if n % 2 == 0:
    median_x = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
    print(f"\n  n = {n} (even)")
    print(f"  median = (x_{n//2} + x_{n//2 + 1}) / 2")
    print(f"  median = ({sorted_data[n//2 - 1]:.0f} + {sorted_data[n//2]:.0f}) / 2")
else:
    median_x = sorted_data[n//2]
    print(f"\n  n = {n} (odd)")
    print(f"  median = x_{(n+1)//2}")
    
print(f"  median = {median_x:.2f}")

print("\n" + "-" * 50)
print("MODE")
print("-" * 50)

print("\n  The mode is the most frequently occurring value.")

# Count occurrences
unique, counts = np.unique(triplet_data, return_counts=True)
max_count = counts.max()
modes = unique[counts == max_count]

print(f"\n  Value frequency distribution:")
for val, cnt in sorted(zip(unique, counts), key=lambda x: -x[1])[:5]:
    print(f"    {val:.0f}: appears {cnt} time(s)")

if len(modes) == 1:
    mode_x = modes[0]
    print(f"\n  Mode = {mode_x:.0f} (appears {max_count} times)")
else:
    print(f"\n  Dataset is multimodal with values: {modes}")
    mode_x = modes[0]

print("\n" + "-" * 50)
print("COMPARISON OF CENTRAL TENDENCY MEASURES")
print("-" * 50)

print(f"\n  Mean:   {mean_x:.2f}")
print(f"  Median: {median_x:.2f}")
print(f"  Mode:   {mode_x:.0f}")

if mean_x > median_x:
    print("\n  Mean > Median suggests positive skewness")
elif mean_x < median_x:
    print("\n  Mean < Median suggests negative skewness")
else:
    print("\n  Mean ≈ Median suggests symmetric distribution")

# =============================================================================
# STEP 3: MEASURES OF DISPERSION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: MEASURES OF DISPERSION")
print("=" * 70)

print("\n" + "-" * 50)
print("RANGE")
print("-" * 50)

print("\n  Formula: Range = x_max - x_min")

x_min = triplet_data.min()
x_max = triplet_data.max()
range_x = x_max - x_min

print(f"\n  x_min = {x_min:.0f}")
print(f"  x_max = {x_max:.0f}")
print(f"  Range = {x_max:.0f} - {x_min:.0f} = {range_x:.0f}")

print("\n" + "-" * 50)
print("VARIANCE AND STANDARD DEVIATION")
print("-" * 50)

print("\n  Population Variance:")
print("  σ² = (1/n) × Σ(xᵢ - μ)²")
print("\n  Sample Variance (unbiased estimator):")
print("  s² = (1/(n-1)) × Σ(xᵢ - x̄)²")
print("\n  Standard Deviation:")
print("  s = √s²")

# Step-by-step calculation
deviations = triplet_data - mean_x
squared_deviations = deviations ** 2
sum_sq_dev = np.sum(squared_deviations)

population_variance = sum_sq_dev / n
sample_variance = sum_sq_dev / (n - 1)
sample_std = np.sqrt(sample_variance)

print(f"\n  Step-by-step calculation:")
print(f"\n  1. Calculate deviations from mean (xᵢ - x̄):")
print(f"     First 5 deviations: {deviations[:5]}")
print(f"\n  2. Square the deviations (xᵢ - x̄)²:")
print(f"     First 5 squared deviations: {squared_deviations[:5]}")
print(f"\n  3. Sum of squared deviations:")
print(f"     Σ(xᵢ - x̄)² = {sum_sq_dev:.2f}")
print(f"\n  4. Population Variance:")
print(f"     σ² = {sum_sq_dev:.2f} / {n} = {population_variance:.2f}")
print(f"\n  5. Sample Variance (using n-1 for unbiased estimation):")
print(f"     s² = {sum_sq_dev:.2f} / {n - 1} = {sample_variance:.2f}")
print(f"\n  6. Sample Standard Deviation:")
print(f"     s = √{sample_variance:.2f} = {sample_std:.2f}")

print("\n" + "-" * 50)
print("INTERQUARTILE RANGE (IQR)")
print("-" * 50)

print("\n  The IQR measures the spread of the middle 50% of data.")
print("  Formula: IQR = Q3 - Q1")

# Calculate quartiles
Q1 = np.percentile(triplet_data, 25)
Q2 = np.percentile(triplet_data, 50)  # Same as median
Q3 = np.percentile(triplet_data, 75)
IQR = Q3 - Q1

print(f"\n  Q1 (25th percentile) = {Q1:.2f}")
print(f"  Q2 (50th percentile / median) = {Q2:.2f}")
print(f"  Q3 (75th percentile) = {Q3:.2f}")
print(f"  IQR = {Q3:.2f} - {Q1:.2f} = {IQR:.2f}")

# Outlier detection using IQR
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outliers = triplet_data[(triplet_data < lower_fence) | (triplet_data > upper_fence)]

print(f"\n  Outlier Detection (1.5 × IQR rule):")
print(f"  Lower fence = Q1 - 1.5×IQR = {Q1:.2f} - 1.5×{IQR:.2f} = {lower_fence:.2f}")
print(f"  Upper fence = Q3 + 1.5×IQR = {Q3:.2f} + 1.5×{IQR:.2f} = {upper_fence:.2f}")
print(f"  Number of outliers: {len(outliers)}")

print("\n" + "-" * 50)
print("COEFFICIENT OF VARIATION")
print("-" * 50)

print("\n  Formula: CV = (s / x̄) × 100%")
print("  The CV expresses standard deviation as a percentage of the mean.")

CV = (sample_std / mean_x) * 100
print(f"\n  CV = ({sample_std:.2f} / {mean_x:.2f}) × 100% = {CV:.2f}%")
print(f"\n  Interpretation: {'High' if CV > 30 else 'Moderate' if CV > 15 else 'Low'} variability")

# =============================================================================
# STEP 4: SHAPE MEASURES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: SHAPE MEASURES")
print("=" * 70)

print("\n" + "-" * 50)
print("SKEWNESS")
print("-" * 50)

print("\n  Formula: γ₁ = (1/n) × Σ[(xᵢ - x̄)/s]³")
print("\n  Skewness measures the asymmetry of the distribution:")
print("  - γ₁ > 0: Positive skew (right tail longer)")
print("  - γ₁ < 0: Negative skew (left tail longer)")
print("  - γ₁ = 0: Symmetric distribution")

# Calculate skewness
standardized = (triplet_data - mean_x) / sample_std
skewness = np.mean(standardized ** 3)

print(f"\n  Calculation:")
print(f"  Standardized values: z = (x - x̄)/s")
print(f"  Skewness γ₁ = {skewness:.4f}")

if abs(skewness) < 0.5:
    interpretation = "approximately symmetric"
elif skewness > 0:
    interpretation = "positively skewed (right-tailed)"
else:
    interpretation = "negatively skewed (left-tailed)"
print(f"  Interpretation: Distribution is {interpretation}")

print("\n" + "-" * 50)
print("KURTOSIS")
print("-" * 50)

print("\n  Formula: γ₂ = (1/n) × Σ[(xᵢ - x̄)/s]⁴ - 3")
print("\n  Kurtosis measures the 'tailedness' of the distribution:")
print("  - γ₂ > 0: Leptokurtic (heavier tails than normal)")
print("  - γ₂ < 0: Platykurtic (lighter tails than normal)")
print("  - γ₂ = 0: Mesokurtic (similar to normal)")

# Calculate excess kurtosis
kurtosis = np.mean(standardized ** 4) - 3

print(f"\n  Excess Kurtosis γ₂ = {kurtosis:.4f}")

if abs(kurtosis) < 0.5:
    interpretation = "mesokurtic (similar to normal)"
elif kurtosis > 0:
    interpretation = "leptokurtic (heavy tails)"
else:
    interpretation = "platykurtic (light tails)"
print(f"  Interpretation: Distribution is {interpretation}")

# =============================================================================
# STEP 5: FIVE-NUMBER SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: FIVE-NUMBER SUMMARY")
print("=" * 70)

print("\n  The five-number summary provides a complete overview of the distribution:")

five_num = {
    'Minimum': x_min,
    'Q1 (25%)': Q1,
    'Median (50%)': Q2,
    'Q3 (75%)': Q3,
    'Maximum': x_max
}

print("\n  ┌─────────────────────────────────────────┐")
print("  │       FIVE-NUMBER SUMMARY               │")
print("  ├─────────────────────────────────────────┤")
for name, value in five_num.items():
    print(f"  │  {name:<20} {value:>10.2f}      │")
print("  └─────────────────────────────────────────┘")

# =============================================================================
# STEP 6: COMPLETE SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: COMPLETE STATISTICS SUMMARY")
print("=" * 70)

summary = {
    'Statistic': [
        'Count (n)', 'Sum (Σxᵢ)', 'Mean (x̄)', 'Median', 'Mode',
        'Minimum', 'Maximum', 'Range',
        'Variance (s²)', 'Std Dev (s)', 'Std Error (SE)',
        'Q1', 'Q3', 'IQR',
        'CV (%)', 'Skewness', 'Kurtosis'
    ],
    'Value': [
        n, sum_x, mean_x, median_x, mode_x,
        x_min, x_max, range_x,
        sample_variance, sample_std, sample_std / np.sqrt(n),
        Q1, Q3, IQR,
        CV, skewness, kurtosis
    ]
}

df_summary = pd.DataFrame(summary)
print("\n")
print(df_summary.to_string(index=False))

# =============================================================================
# STEP 7: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histogram with central tendency measures
ax1 = axes[0, 0]
ax1.hist(triplet_data, bins=15, color='steelblue', edgecolor='white', 
         alpha=0.7, density=True, label='Data')
ax1.axvline(mean_x, color='red', linestyle='-', linewidth=2.5, 
            label=f'Mean = {mean_x:.1f}')
ax1.axvline(median_x, color='green', linestyle='--', linewidth=2.5, 
            label=f'Median = {median_x:.1f}')
ax1.axvline(mode_x, color='orange', linestyle=':', linewidth=2.5, 
            label=f'Mode = {mode_x:.0f}')
ax1.set_xlabel('Triplet Deliveries per Year', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('Distribution with Central Tendency Measures', 
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Box plot with annotations
ax2 = axes[0, 1]
bp = ax2.boxplot(triplet_data, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('steelblue')
bp['boxes'][0].set_alpha(0.7)
ax2.set_ylabel('Triplet Deliveries per Year', fontsize=11)
ax2.set_title('Box Plot with Five-Number Summary', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Add annotations
ax2.annotate(f'Max: {x_max:.0f}', xy=(1.15, x_max), fontsize=9)
ax2.annotate(f'Q3: {Q3:.0f}', xy=(1.15, Q3), fontsize=9)
ax2.annotate(f'Median: {Q2:.0f}', xy=(1.15, Q2), fontsize=9)
ax2.annotate(f'Q1: {Q1:.0f}', xy=(1.15, Q1), fontsize=9)
ax2.annotate(f'Min: {x_min:.0f}', xy=(1.15, x_min), fontsize=9)

# Plot 3: Time series with mean and standard deviation bands
ax3 = axes[1, 0]
ax3.plot(years, triplet_data, 'o-', color='steelblue', 
         linewidth=2, markersize=6, label='Triplet Births')
ax3.axhline(mean_x, color='red', linestyle='-', linewidth=2, 
            label=f'Mean = {mean_x:.1f}')
ax3.fill_between(years, mean_x - sample_std, mean_x + sample_std, 
                 alpha=0.2, color='red', label=f'±1σ (±{sample_std:.1f})')
ax3.fill_between(years, mean_x - 2*sample_std, mean_x + 2*sample_std, 
                 alpha=0.1, color='red', label=f'±2σ (±{2*sample_std:.1f})')
ax3.set_xlabel('Year', fontsize=11)
ax3.set_ylabel('Triplet Deliveries', fontsize=11)
ax3.set_title('Time Series with Mean and Standard Deviation Bands', 
              fontsize=12, fontweight='bold')
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(True, alpha=0.3)

# Plot 4: QQ plot for normality assessment
ax4 = axes[1, 1]
sorted_data = np.sort(triplet_data)
theoretical_quantiles = np.linspace(0.5/n, 1-0.5/n, n)
from scipy.stats import norm
theoretical_values = norm.ppf(theoretical_quantiles) * sample_std + mean_x

ax4.scatter(theoretical_values, sorted_data, color='steelblue', 
            edgecolors='white', s=50, alpha=0.7)
ax4.plot([min(theoretical_values), max(theoretical_values)], 
         [min(theoretical_values), max(theoretical_values)], 
         'r--', linewidth=2, label='Perfect Normal')
ax4.set_xlabel('Theoretical Quantiles', fontsize=11)
ax4.set_ylabel('Sample Quantiles', fontsize=11)
ax4.set_title('Q-Q Plot (Normality Assessment)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
# Ensure output directory exists
output_dir = os.path.join(script_dir, '../outputs/figures')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(script_dir, '../outputs/figures/descriptive_statistics_triplet_births.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: descriptive_statistics_triplet_births.png")

# =============================================================================
# STEP 8: KEY FINDINGS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: KEY FINDINGS")
print("=" * 70)

print("\n--- Central Tendency ---")
print(f"  Mean triplet births per year: {mean_x:.1f}")
print(f"  The median ({median_x:.1f}) is {'close to' if abs(mean_x - median_x) < 5 else 'different from'} the mean")

print("\n--- Dispersion ---")
print(f"  Standard deviation: {sample_std:.2f}")
print(f"  68% of years have between {mean_x - sample_std:.0f} and {mean_x + sample_std:.0f} triplet births")
print(f"  95% of years have between {mean_x - 2*sample_std:.0f} and {mean_x + 2*sample_std:.0f} triplet births")

print("\n--- Distribution Shape ---")
print(f"  Skewness: {skewness:.4f} ({interpretation})")
print(f"  The data appears to follow a fairly symmetric distribution")

print("\n--- Dataset Reference ---")
print("  Human Multiple Births Database (2024) FRA_InputData_25.11.2024.xlsx.")
print("  Available at: https://www.twinbirths.org/en/data-metadata/")
print("  (Accessed: 25 November 2024).")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF DESCRIPTIVE STATISTICS ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
