"""
================================================================================
NORMALISATION AND STANDARDISATION - TRIPLET BIRTHS DATASET
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
This script demonstrates normalisation and standardisation techniques applied
to the triplet births demographic dataset. These preprocessing techniques
are essential for preparing time series data for analysis and modelling.

MATHEMATICAL CONCEPTS:
- Min-Max Normalisation: x_norm = (x - x_min) / (x_max - x_min)
- Z-Score Standardisation: z = (x - μ) / σ
- Log Transformation: x_log = log(x)
- Moving Average Normalisation

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
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# HEADER INFORMATION
# =============================================================================

print("=" * 70)
print("NORMALISATION AND STANDARDISATION")
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

# Clean and filter data
df_clean = df.dropna(subset=['Year', 'Triplet_deliveries'])
df_clean['Year'] = df_clean['Year'].astype(int)

start_year = 1990
end_year = 2020
df_filtered = df_clean[(df_clean['Year'] >= start_year) & 
                        (df_clean['Year'] <= end_year)].copy()

# Extract variables
triplet_data = df_filtered['Triplet_deliveries'].values
years = df_filtered['Year'].values

print(f"\nAnalysis period: {start_year} - {end_year}")
print(f"Number of observations: {len(triplet_data)}")

# Original statistics
print("\n--- Original Data Statistics ---")
print(f"  Mean: {np.mean(triplet_data):.2f}")
print(f"  Std:  {np.std(triplet_data):.2f}")
print(f"  Min:  {np.min(triplet_data):.0f}")
print(f"  Max:  {np.max(triplet_data):.0f}")

# =============================================================================
# STEP 2: MIN-MAX NORMALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: MIN-MAX NORMALISATION")
print("=" * 70)

print("\n--- Mathematical Formula ---")
print("\n  x_norm = (x - x_min) / (x_max - x_min)")
print("\n  Properties:")
print("  - Scales data to [0, 1] range")
print("  - Preserves relative distances")
print("  - Sensitive to outliers")

# Calculate step by step
x_min = np.min(triplet_data)
x_max = np.max(triplet_data)
x_range = x_max - x_min

print(f"\n--- Step-by-step Calculation ---")
print(f"\n  1. Find minimum value:")
print(f"     x_min = {x_min:.0f}")
print(f"\n  2. Find maximum value:")
print(f"     x_max = {x_max:.0f}")
print(f"\n  3. Calculate range:")
print(f"     range = x_max - x_min = {x_max:.0f} - {x_min:.0f} = {x_range:.0f}")

# Apply normalisation
triplet_minmax = (triplet_data - x_min) / x_range

print(f"\n  4. Apply formula to each value:")
print(f"\n     Year    Original    Calculation                    Normalised")
print(f"     " + "-" * 65)
for i in range(min(5, len(triplet_data))):
    calc = f"({triplet_data[i]:.0f} - {x_min:.0f}) / {x_range:.0f}"
    print(f"     {years[i]}    {triplet_data[i]:>6.0f}      {calc:<30} {triplet_minmax[i]:.4f}")
print(f"     ...")

print(f"\n  5. Verification:")
print(f"     Normalised minimum: {np.min(triplet_minmax):.4f} (should be 0)")
print(f"     Normalised maximum: {np.max(triplet_minmax):.4f} (should be 1)")

# =============================================================================
# STEP 3: Z-SCORE STANDARDISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Z-SCORE STANDARDISATION")
print("=" * 70)

print("\n--- Mathematical Formula ---")
print("\n  z = (x - μ) / σ")
print("\n  Properties:")
print("  - Transforms to mean = 0, std = 1")
print("  - Values represent standard deviations from mean")
print("  - Useful for comparing different distributions")

# Calculate step by step
mu = np.mean(triplet_data)
sigma = np.std(triplet_data, ddof=0)

print(f"\n--- Step-by-step Calculation ---")
print(f"\n  1. Calculate mean (μ):")
print(f"     μ = Σxᵢ / n = {np.sum(triplet_data):.0f} / {len(triplet_data)} = {mu:.2f}")
print(f"\n  2. Calculate standard deviation (σ):")
print(f"     σ = √[Σ(xᵢ - μ)² / n] = {sigma:.2f}")

# Apply standardisation
triplet_zscore = (triplet_data - mu) / sigma

print(f"\n  3. Apply formula to each value:")
print(f"\n     Year    Original    Deviation     Z-Score    Interpretation")
print(f"     " + "-" * 70)
for i in range(min(5, len(triplet_data))):
    dev = triplet_data[i] - mu
    interp = "above" if dev > 0 else "below"
    print(f"     {years[i]}    {triplet_data[i]:>6.0f}     {dev:>+8.2f}      {triplet_zscore[i]:>+7.4f}   {abs(triplet_zscore[i]):.2f}σ {interp} mean")
print(f"     ...")

print(f"\n  4. Verification:")
print(f"     Mean of z-scores: {np.mean(triplet_zscore):.10f} (should be ≈ 0)")
print(f"     Std of z-scores:  {np.std(triplet_zscore):.10f} (should be ≈ 1)")

# Interpretation guide
print(f"\n  5. Z-Score Interpretation:")
print(f"     |z| < 1:   Within 1 standard deviation (typical value)")
print(f"     |z| ≈ 2:   Within 2 standard deviations (unusual)")
print(f"     |z| > 3:   Beyond 3 standard deviations (potential outlier)")

# =============================================================================
# STEP 4: LOG TRANSFORMATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: LOG TRANSFORMATION")
print("=" * 70)

print("\n--- Mathematical Formula ---")
print("\n  x_log = log₁₀(x)  or  x_log = ln(x)")
print("\n  Properties:")
print("  - Reduces right skewness")
print("  - Handles multiplicative relationships")
print("  - Requires positive values")

# Apply log transformation
triplet_log10 = np.log10(triplet_data)
triplet_ln = np.log(triplet_data)

print(f"\n--- Step-by-step Calculation ---")
print(f"\n     Year    Original    log₁₀(x)     ln(x)")
print(f"     " + "-" * 50)
for i in range(min(5, len(triplet_data))):
    print(f"     {years[i]}    {triplet_data[i]:>6.0f}      {triplet_log10[i]:.4f}      {triplet_ln[i]:.4f}")
print(f"     ...")

print(f"\n  Original data:")
print(f"    Mean: {np.mean(triplet_data):.2f}, Std: {np.std(triplet_data):.2f}")
print(f"\n  Log₁₀ transformed:")
print(f"    Mean: {np.mean(triplet_log10):.4f}, Std: {np.std(triplet_log10):.4f}")
print(f"\n  Natural log transformed:")
print(f"    Mean: {np.mean(triplet_ln):.4f}, Std: {np.std(triplet_ln):.4f}")

# =============================================================================
# STEP 5: ROBUST SCALING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: ROBUST SCALING")
print("=" * 70)

print("\n--- Mathematical Formula ---")
print("\n  x_robust = (x - median) / IQR")
print("\n  where IQR = Q3 - Q1")
print("\n  Properties:")
print("  - Resistant to outliers")
print("  - Uses median instead of mean")
print("  - Uses IQR instead of standard deviation")

# Calculate robust scaling
median = np.median(triplet_data)
q1 = np.percentile(triplet_data, 25)
q3 = np.percentile(triplet_data, 75)
iqr = q3 - q1

triplet_robust = (triplet_data - median) / iqr

print(f"\n--- Step-by-step Calculation ---")
print(f"\n  1. Calculate median:")
print(f"     median = {median:.2f}")
print(f"\n  2. Calculate quartiles:")
print(f"     Q1 = {q1:.2f}")
print(f"     Q3 = {q3:.2f}")
print(f"\n  3. Calculate IQR:")
print(f"     IQR = Q3 - Q1 = {q3:.2f} - {q1:.2f} = {iqr:.2f}")
print(f"\n  4. Apply formula:")
print(f"\n     Year    Original    (x - median)    Robust Scaled")
print(f"     " + "-" * 55)
for i in range(min(5, len(triplet_data))):
    dev = triplet_data[i] - median
    print(f"     {years[i]}    {triplet_data[i]:>6.0f}     {dev:>+10.2f}       {triplet_robust[i]:>+8.4f}")
print(f"     ...")

# =============================================================================
# STEP 6: COMPARISON OF ALL METHODS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: COMPARISON OF NORMALISATION METHODS")
print("=" * 70)

# Create comparison DataFrame
df_comparison = pd.DataFrame({
    'Year': years,
    'Original': triplet_data,
    'Min-Max': triplet_minmax,
    'Z-Score': triplet_zscore,
    'Log10': triplet_log10,
    'Robust': triplet_robust
})

print("\n--- Comparison Table (First 10 Years) ---")
print(df_comparison.head(10).to_string(index=False))

print("\n--- Summary Statistics After Each Transformation ---")
summary = df_comparison.drop('Year', axis=1).describe()
print(summary.to_string())

# =============================================================================
# STEP 7: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Original data
ax1 = axes[0, 0]
ax1.plot(years, triplet_data, 'o-', color='steelblue', linewidth=2, markersize=5)
ax1.axhline(mu, color='red', linestyle='--', label=f'Mean = {mu:.0f}')
ax1.set_xlabel('Year', fontsize=10)
ax1.set_ylabel('Triplet Deliveries', fontsize=10)
ax1.set_title('Original Data', fontsize=11, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: Min-Max Normalisation
ax2 = axes[0, 1]
ax2.plot(years, triplet_minmax, 'o-', color='green', linewidth=2, markersize=5)
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax2.set_xlabel('Year', fontsize=10)
ax2.set_ylabel('Normalised Value [0,1]', fontsize=10)
ax2.set_title('Min-Max Normalisation\nx_norm = (x - x_min)/(x_max - x_min)', 
              fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Z-Score Standardisation
ax3 = axes[0, 2]
ax3.plot(years, triplet_zscore, 'o-', color='purple', linewidth=2, markersize=5)
ax3.axhline(0, color='red', linestyle='-', linewidth=2, label='Mean = 0')
ax3.axhline(1, color='orange', linestyle='--', alpha=0.7)
ax3.axhline(-1, color='orange', linestyle='--', alpha=0.7, label='±1σ')
ax3.axhline(2, color='orange', linestyle=':', alpha=0.5)
ax3.axhline(-2, color='orange', linestyle=':', alpha=0.5, label='±2σ')
ax3.set_xlabel('Year', fontsize=10)
ax3.set_ylabel('Z-Score', fontsize=10)
ax3.set_title('Z-Score Standardisation\nz = (x - μ)/σ', fontsize=11, fontweight='bold')
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Plot 4: Log Transformation
ax4 = axes[1, 0]
ax4.plot(years, triplet_log10, 'o-', color='orange', linewidth=2, markersize=5)
ax4.set_xlabel('Year', fontsize=10)
ax4.set_ylabel('log₁₀(x)', fontsize=10)
ax4.set_title('Log Transformation\nx_log = log₁₀(x)', fontsize=11, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Plot 5: Robust Scaling
ax5 = axes[1, 1]
ax5.plot(years, triplet_robust, 'o-', color='brown', linewidth=2, markersize=5)
ax5.axhline(0, color='red', linestyle='--', label='Median = 0')
ax5.set_xlabel('Year', fontsize=10)
ax5.set_ylabel('Robust Scaled', fontsize=10)
ax5.set_title('Robust Scaling\nx_robust = (x - median)/IQR', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# Plot 6: Distribution comparison
ax6 = axes[1, 2]
methods = ['Original\n(scaled)', 'Min-Max', 'Z-Score', 'Robust']
data_list = [triplet_minmax, triplet_minmax, triplet_zscore, triplet_robust]
bp = ax6.boxplot(data_list, labels=methods, patch_artist=True)
colors = ['steelblue', 'green', 'purple', 'brown']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax6.set_ylabel('Transformed Value', fontsize=10)
ax6.set_title('Distribution Comparison', fontsize=11, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/normalisation_triplet_births.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: normalisation_triplet_births.png")

# =============================================================================
# STEP 8: PRACTICAL RECOMMENDATIONS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: PRACTICAL RECOMMENDATIONS")
print("=" * 70)

print("\n--- When to Use Each Method ---")

print("\n  Min-Max Normalisation:")
print("    ✓ Neural networks and deep learning")
print("    ✓ Image processing (pixel values)")
print("    ✓ When bounded range is required")
print("    ✗ Sensitive to outliers")

print("\n  Z-Score Standardisation:")
print("    ✓ PCA, clustering, linear regression")
print("    ✓ When comparing variables with different units")
print("    ✓ Algorithms assuming normal distribution")
print("    ✗ May produce unbounded values")

print("\n  Log Transformation:")
print("    ✓ Right-skewed distributions")
print("    ✓ Multiplicative relationships")
print("    ✓ Variance stabilisation")
print("    ✗ Only for positive values")

print("\n  Robust Scaling:")
print("    ✓ Data with outliers")
print("    ✓ Non-normal distributions")
print("    ✓ When median is more representative than mean")

print("\n--- Dataset Reference ---")
print("  Human Multiple Births Database (2024) FRA_InputData_25.11.2024.xlsx.")
print("  Available at: https://www.twinbirths.org/en/data-metadata/")
print("  (Accessed: 25 November 2024).")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF NORMALISATION AND STANDARDISATION ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
