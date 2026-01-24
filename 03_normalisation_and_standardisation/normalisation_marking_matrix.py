"""
================================================================================
NORMALISATION AND STANDARDISATION - MARKING MATRIX DATASET
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
to numerical features derived from the marking matrix dataset. These
transformations are essential for data preprocessing in machine learning.

MATHEMATICAL CONCEPTS:
- Min-Max Normalisation: x_norm = (x - x_min) / (x_max - x_min)
- Z-Score Standardisation: z = (x - μ) / σ
- Decimal Scaling: x_scaled = x / 10^j
- Robust Scaling: x_scaled = (x - median) / IQR

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
# STEP 1: LOAD THE DATASET AND CREATE FEATURES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING DATA AND CREATING FEATURES")
print("=" * 70)

file_path = '../datasets/COM7023_Mathematics_for_Data_Science_Marking_Matrix.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("\nDataset loaded successfully!")
except FileNotFoundError:
    print("\nError: Dataset file not found.")
    exit()

# Create numerical features from text data
grade_columns = ['90- 100', '80 - 90', '70 – 79', '60 - 69', 
                 '50 – 59', '40 – 49', '30 – 39', '20 - 29', '0 - 19']

grade_midpoints = {
    '90- 100': 95, '80 - 90': 85, '70 – 79': 74.5, '60 - 69': 64.5,
    '50 – 59': 54.5, '40 – 49': 44.5, '30 – 39': 34.5, '20 - 29': 24.5, '0 - 19': 9.5
}

text_data = []
for idx, row in df.iterrows():
    for grade_col in grade_columns:
        if grade_col in df.columns:
            text = str(row[grade_col])
            text_data.append({
                'LO': idx + 1,
                'Grade_Band': grade_col,
                'Grade_Midpoint': grade_midpoints[grade_col],
                'Word_Count': len(text.split()),
                'Character_Count': len(text)
            })

df_features = pd.DataFrame(text_data)
print(f"\nCreated feature dataset with {len(df_features)} observations")

# Display original statistics
print("\n--- Original Data Statistics ---")
print(df_features[['Grade_Midpoint', 'Word_Count', 'Character_Count']].describe())

# =============================================================================
# STEP 2: MIN-MAX NORMALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: MIN-MAX NORMALISATION")
print("=" * 70)

print("\n--- Mathematical Formula ---")
print("\n  x_norm = (x - x_min) / (x_max - x_min)")
print("\n  This scales all values to the range [0, 1]")
print("  - Minimum value maps to 0")
print("  - Maximum value maps to 1")

def min_max_normalise(data):
    """
    Apply Min-Max normalisation to data.
    
    Parameters:
        data: array-like, original values
        
    Returns:
        normalised: array, values scaled to [0, 1]
        x_min: minimum value
        x_max: maximum value
    """
    x_min = np.min(data)
    x_max = np.max(data)
    normalised = (data - x_min) / (x_max - x_min)
    return normalised, x_min, x_max

# Apply to Word_Count
word_counts = df_features['Word_Count'].values
word_norm, word_min, word_max = min_max_normalise(word_counts)

print("\n--- Word Count Normalisation ---")
print(f"\n  Step-by-step calculation:")
print(f"  x_min = {word_min}")
print(f"  x_max = {word_max}")
print(f"\n  Example: For x = {word_counts[0]}")
print(f"  x_norm = ({word_counts[0]} - {word_min}) / ({word_max} - {word_min})")
print(f"  x_norm = {word_counts[0] - word_min} / {word_max - word_min}")
print(f"  x_norm = {word_norm[0]:.4f}")

print(f"\n  Original range: [{word_min}, {word_max}]")
print(f"  Normalised range: [{word_norm.min():.4f}, {word_norm.max():.4f}]")

# Add normalised column
df_features['Word_Count_MinMax'] = word_norm

# =============================================================================
# STEP 3: Z-SCORE STANDARDISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: Z-SCORE STANDARDISATION")
print("=" * 70)

print("\n--- Mathematical Formula ---")
print("\n  z = (x - μ) / σ")
print("\n  where:")
print("  μ = population mean")
print("  σ = population standard deviation")
print("\n  This transforms data to have:")
print("  - Mean = 0")
print("  - Standard deviation = 1")

def z_score_standardise(data):
    """
    Apply Z-score standardisation to data.
    
    Parameters:
        data: array-like, original values
        
    Returns:
        standardised: array, z-scores
        mean: mean of original data
        std: standard deviation of original data
    """
    mean = np.mean(data)
    std = np.std(data, ddof=0)  # Population std
    standardised = (data - mean) / std
    return standardised, mean, std

# Apply to Word_Count
word_z, word_mean, word_std = z_score_standardise(word_counts)

print("\n--- Word Count Standardisation ---")
print(f"\n  Step-by-step calculation:")
print(f"  μ = {word_mean:.4f}")
print(f"  σ = {word_std:.4f}")
print(f"\n  Example: For x = {word_counts[0]}")
print(f"  z = ({word_counts[0]} - {word_mean:.4f}) / {word_std:.4f}")
print(f"  z = {word_counts[0] - word_mean:.4f} / {word_std:.4f}")
print(f"  z = {word_z[0]:.4f}")

print(f"\n  Verification:")
print(f"  Mean of z-scores: {word_z.mean():.10f} (should be ≈ 0)")
print(f"  Std of z-scores: {word_z.std():.10f} (should be ≈ 1)")

# Add standardised column
df_features['Word_Count_ZScore'] = word_z

# =============================================================================
# STEP 4: DECIMAL SCALING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: DECIMAL SCALING")
print("=" * 70)

print("\n--- Mathematical Formula ---")
print("\n  x_scaled = x / 10^j")
print("\n  where j is the smallest integer such that max(|x_scaled|) < 1")

def decimal_scaling(data):
    """
    Apply decimal scaling to data.
    
    Parameters:
        data: array-like, original values
        
    Returns:
        scaled: array, decimal scaled values
        j: power of 10 used
    """
    max_abs = np.max(np.abs(data))
    j = np.ceil(np.log10(max_abs + 1))  # +1 to handle exact powers
    scaled = data / (10 ** j)
    return scaled, int(j)

# Apply to Character_Count
char_counts = df_features['Character_Count'].values
char_scaled, j = decimal_scaling(char_counts)

print("\n--- Character Count Decimal Scaling ---")
print(f"\n  Step-by-step calculation:")
print(f"  max(|x|) = {np.max(np.abs(char_counts))}")
print(f"  j = ceil(log₁₀({np.max(np.abs(char_counts))})) = {j}")
print(f"\n  Example: For x = {char_counts[0]}")
print(f"  x_scaled = {char_counts[0]} / 10^{j}")
print(f"  x_scaled = {char_counts[0]} / {10**j}")
print(f"  x_scaled = {char_scaled[0]:.6f}")

print(f"\n  Scaled range: [{char_scaled.min():.6f}, {char_scaled.max():.6f}]")

# Add scaled column
df_features['Character_Count_Decimal'] = char_scaled

# =============================================================================
# STEP 5: ROBUST SCALING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: ROBUST SCALING (MEDIAN AND IQR)")
print("=" * 70)

print("\n--- Mathematical Formula ---")
print("\n  x_robust = (x - median) / IQR")
print("\n  where:")
print("  median = Q2 (50th percentile)")
print("  IQR = Q3 - Q1 (interquartile range)")
print("\n  This method is robust to outliers")

def robust_scaling(data):
    """
    Apply robust scaling using median and IQR.
    
    Parameters:
        data: array-like, original values
        
    Returns:
        scaled: array, robust scaled values
        median: median of data
        iqr: interquartile range
    """
    median = np.median(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    scaled = (data - median) / iqr
    return scaled, median, iqr

# Apply to Word_Count
word_robust, word_median, word_iqr = robust_scaling(word_counts)

print("\n--- Word Count Robust Scaling ---")
print(f"\n  Step-by-step calculation:")
print(f"  median = {word_median:.2f}")
print(f"  Q1 = {np.percentile(word_counts, 25):.2f}")
print(f"  Q3 = {np.percentile(word_counts, 75):.2f}")
print(f"  IQR = {word_iqr:.2f}")
print(f"\n  Example: For x = {word_counts[0]}")
print(f"  x_robust = ({word_counts[0]} - {word_median:.2f}) / {word_iqr:.2f}")
print(f"  x_robust = {word_robust[0]:.4f}")

# Add robust scaled column
df_features['Word_Count_Robust'] = word_robust

# =============================================================================
# STEP 6: COMPARISON OF ALL METHODS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: COMPARISON OF NORMALISATION METHODS")
print("=" * 70)

# Create comparison table
comparison = pd.DataFrame({
    'Original': word_counts,
    'Min-Max [0,1]': word_norm,
    'Z-Score': word_z,
    'Robust': word_robust
})

print("\n--- First 10 Observations ---")
print(comparison.head(10).to_string())

print("\n--- Summary Statistics After Transformation ---")
summary = comparison.describe()
print(summary.to_string())

# =============================================================================
# STEP 7: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Original vs Min-Max
ax1 = axes[0, 0]
ax1.scatter(word_counts, word_norm, alpha=0.6, color='steelblue', 
            edgecolors='white', s=50)
ax1.set_xlabel('Original Word Count', fontsize=11)
ax1.set_ylabel('Min-Max Normalised', fontsize=11)
ax1.set_title('Min-Max Normalisation\nx_norm = (x - x_min) / (x_max - x_min)', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Distribution comparison
ax2 = axes[0, 1]
ax2.hist(word_counts, bins=15, alpha=0.5, label='Original', color='blue')
ax2.set_xlabel('Value', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Original Distribution', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Z-score distribution
ax3 = axes[1, 0]
ax3.hist(word_z, bins=15, alpha=0.7, color='green', edgecolor='white')
ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='Mean = 0')
ax3.axvline(-1, color='orange', linestyle=':', linewidth=1.5, label='±1σ')
ax3.axvline(1, color='orange', linestyle=':', linewidth=1.5)
ax3.set_xlabel('Z-Score', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Z-Score Standardisation\nz = (x - μ) / σ', 
              fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Box plots comparison
ax4 = axes[1, 1]
data_to_plot = [word_counts, word_norm * 100, word_z]  # Scale min-max for visibility
bp = ax4.boxplot(data_to_plot, labels=['Original', 'Min-Max (×100)', 'Z-Score'], 
                 patch_artist=True)
colors = ['steelblue', 'lightgreen', 'salmon']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Value', fontsize=11)
ax4.set_title('Comparison of Scaling Methods', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/normalisation_marking_matrix.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: normalisation_marking_matrix.png")

# =============================================================================
# STEP 8: PRACTICAL APPLICATIONS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: WHEN TO USE EACH METHOD")
print("=" * 70)

print("\n--- Min-Max Normalisation ---")
print("  Use when:")
print("  - You need values in a specific range [0, 1]")
print("  - Data does not have significant outliers")
print("  - For neural networks and image processing")

print("\n--- Z-Score Standardisation ---")
print("  Use when:")
print("  - Data is approximately normally distributed")
print("  - For algorithms assuming normal distribution (e.g., PCA, LDA)")
print("  - When comparing variables with different units")

print("\n--- Decimal Scaling ---")
print("  Use when:")
print("  - You need simple, reversible transformation")
print("  - Data has consistent order of magnitude")

print("\n--- Robust Scaling ---")
print("  Use when:")
print("  - Data contains outliers")
print("  - Distribution is not normal")
print("  - More resistant to extreme values")

print("\n--- Dataset Reference ---")
print("  Arden University (2024) COM7023 Mathematics for Data Science")
print("  Marking Matrix. Arden University.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF NORMALISATION AND STANDARDISATION ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
