"""
================================================================================
DESCRIPTIVE STATISTICS - MARKING MATRIX DATASET
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
This script demonstrates descriptive statistics applied to the marking matrix
dataset. Since this dataset contains primarily text data, we apply statistical
measures to derived numerical features such as text length, word count, and
grade band scores.

MATHEMATICAL CONCEPTS:
- Measures of central tendency: mean, median, mode
- Measures of dispersion: variance, standard deviation, range, IQR
- Shape measures: skewness, kurtosis

DATASET REFERENCE:
Arden University (2024) COM7023 Mathematics for Data Science Marking Matrix.
Arden University.
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
# STEP 1: LOAD THE DATASET
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING THE DATASET")
print("=" * 70)

file_path = os.path.join(os.path.dirname(__file__), '../datasets/COM7023_Mathematics_for_Data_Science_Marking_Matrix.csv')

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("\nDataset loaded successfully!")
except FileNotFoundError:
    print("\nError: Dataset file not found.")
    exit()

# Display basic info
print(f"Dataset shape: {df.shape}")

# =============================================================================
# STEP 2: CREATE NUMERICAL FEATURES FROM TEXT DATA
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 70)

print("\n--- Creating Numerical Features from Text ---")

# Define grade columns
grade_columns = ['90- 100', '80 - 90', '70 – 79', '60 - 69', 
                 '50 – 59', '40 – 49', '30 – 39', '20 - 29', '0 - 19']

# Create midpoint scores for each grade band
grade_midpoints = {
    '90- 100': 95, '80 - 90': 85, '70 – 79': 74.5, '60 - 69': 64.5,
    '50 – 59': 54.5, '40 – 49': 44.5, '30 – 39': 34.5, '20 - 29': 24.5, '0 - 19': 9.5
}

# Calculate text statistics for each cell
text_stats = []

for idx, row in df.iterrows():
    for grade_col in grade_columns:
        if grade_col in df.columns:
            text = str(row[grade_col])
            word_count = len(text.split())
            char_count = len(text)
            sentence_count = text.count('.') + text.count('!') + text.count('?')
            
            text_stats.append({
                'LO': f'LO{idx + 1}',
                'Grade_Band': grade_col,
                'Grade_Midpoint': grade_midpoints[grade_col],
                'Word_Count': word_count,
                'Character_Count': char_count,
                'Sentence_Count': max(1, sentence_count)
            })

df_stats = pd.DataFrame(text_stats)
print("\nCreated features: Word_Count, Character_Count, Sentence_Count")
print(f"Total observations: {len(df_stats)}")

# =============================================================================
# STEP 3: MEASURES OF CENTRAL TENDENCY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: MEASURES OF CENTRAL TENDENCY")
print("=" * 70)

print("\n--- Mathematical Formulas ---")
print("\n  Arithmetic Mean:")
print("  x̄ = (1/n) × Σxᵢ")
print("\n  Median: Middle value when data is ordered")
print("\n  Mode: Most frequently occurring value")

# Calculate for word count
word_counts = df_stats['Word_Count']

# Mean calculation step by step
n = len(word_counts)
sum_x = word_counts.sum()
mean_word = sum_x / n

print("\n--- Word Count Statistics ---")
print(f"\n  Step-by-step Mean Calculation:")
print(f"  n = {n}")
print(f"  Σxᵢ = {sum_x}")
print(f"  x̄ = {sum_x} / {n} = {mean_word:.2f}")

# Median
sorted_words = word_counts.sort_values().values
if n % 2 == 0:
    median_word = (sorted_words[n//2 - 1] + sorted_words[n//2]) / 2
else:
    median_word = sorted_words[n//2]

print(f"\n  Median = {median_word:.2f}")

# Mode
mode_word = word_counts.mode().values[0]
print(f"  Mode = {mode_word}")

# Calculate by grade band
print("\n--- Mean Word Count by Grade Band ---")
mean_by_grade = df_stats.groupby('Grade_Band')['Word_Count'].mean()
for grade in grade_columns:
    if grade in mean_by_grade.index:
        print(f"  {grade}: {mean_by_grade[grade]:.1f} words")

# =============================================================================
# STEP 4: MEASURES OF DISPERSION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: MEASURES OF DISPERSION")
print("=" * 70)

print("\n--- Mathematical Formulas ---")
print("\n  Sample Variance:")
print("  s² = (1/(n-1)) × Σ(xᵢ - x̄)²")
print("\n  Standard Deviation:")
print("  s = √s²")
print("\n  Range:")
print("  Range = xₘₐₓ - xₘᵢₙ")

# Variance calculation step by step
deviations = word_counts - mean_word
squared_deviations = deviations ** 2
sum_squared_dev = squared_deviations.sum()
variance_word = sum_squared_dev / (n - 1)
std_word = np.sqrt(variance_word)

print("\n--- Word Count Dispersion ---")
print(f"\n  Step-by-step Variance Calculation:")
print(f"  x̄ = {mean_word:.2f}")
print(f"  Σ(xᵢ - x̄)² = {sum_squared_dev:.2f}")
print(f"  s² = {sum_squared_dev:.2f} / {n - 1} = {variance_word:.2f}")
print(f"  s = √{variance_word:.2f} = {std_word:.2f}")

# Range
range_word = word_counts.max() - word_counts.min()
print(f"\n  Range = {word_counts.max()} - {word_counts.min()} = {range_word}")

# Interquartile Range
Q1 = word_counts.quantile(0.25)
Q3 = word_counts.quantile(0.75)
IQR = Q3 - Q1

print(f"\n  Q1 (25th percentile) = {Q1:.2f}")
print(f"  Q3 (75th percentile) = {Q3:.2f}")
print(f"  IQR = Q3 - Q1 = {IQR:.2f}")

# Coefficient of Variation
CV = (std_word / mean_word) * 100
print(f"\n  Coefficient of Variation = (s/x̄) × 100 = {CV:.2f}%")

# =============================================================================
# STEP 5: SHAPE MEASURES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: SHAPE MEASURES")
print("=" * 70)

print("\n--- Skewness ---")
print("  Formula: γ₁ = (1/n) × Σ((xᵢ - x̄)/s)³")

# Calculate skewness manually
standardized = (word_counts - mean_word) / std_word
skewness = (standardized ** 3).mean()

print(f"\n  Skewness = {skewness:.4f}")
if skewness > 0:
    print("  Interpretation: Positive skew (right-tailed distribution)")
elif skewness < 0:
    print("  Interpretation: Negative skew (left-tailed distribution)")
else:
    print("  Interpretation: Symmetric distribution")

print("\n--- Kurtosis ---")
print("  Formula: γ₂ = (1/n) × Σ((xᵢ - x̄)/s)⁴ - 3")

kurtosis = (standardized ** 4).mean() - 3

print(f"\n  Excess Kurtosis = {kurtosis:.4f}")
if kurtosis > 0:
    print("  Interpretation: Leptokurtic (heavier tails than normal)")
elif kurtosis < 0:
    print("  Interpretation: Platykurtic (lighter tails than normal)")
else:
    print("  Interpretation: Mesokurtic (similar to normal distribution)")

# =============================================================================
# STEP 6: SUMMARY TABLE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: COMPLETE STATISTICS SUMMARY")
print("=" * 70)

summary_stats = {
    'Statistic': ['Count', 'Mean', 'Median', 'Mode', 'Variance', 
                  'Std Dev', 'Min', 'Max', 'Range', 'Q1', 'Q3', 
                  'IQR', 'Skewness', 'Kurtosis', 'CV (%)'],
    'Word_Count': [n, mean_word, median_word, mode_word, variance_word,
                   std_word, word_counts.min(), word_counts.max(), range_word,
                   Q1, Q3, IQR, skewness, kurtosis, CV],
    'Character_Count': [
        len(df_stats['Character_Count']),
        df_stats['Character_Count'].mean(),
        df_stats['Character_Count'].median(),
        df_stats['Character_Count'].mode().values[0],
        df_stats['Character_Count'].var(),
        df_stats['Character_Count'].std(),
        df_stats['Character_Count'].min(),
        df_stats['Character_Count'].max(),
        df_stats['Character_Count'].max() - df_stats['Character_Count'].min(),
        df_stats['Character_Count'].quantile(0.25),
        df_stats['Character_Count'].quantile(0.75),
        df_stats['Character_Count'].quantile(0.75) - df_stats['Character_Count'].quantile(0.25),
        ((df_stats['Character_Count'] - df_stats['Character_Count'].mean()) / 
         df_stats['Character_Count'].std() ** 3).mean(),
        ((df_stats['Character_Count'] - df_stats['Character_Count'].mean()) / 
         df_stats['Character_Count'].std() ** 4).mean() - 3,
        (df_stats['Character_Count'].std() / df_stats['Character_Count'].mean()) * 100
    ]
}

df_summary = pd.DataFrame(summary_stats)
print("\n")
print(df_summary.to_string(index=False))

# =============================================================================
# STEP 7: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Distribution of word counts
ax1 = axes[0, 0]
ax1.hist(word_counts, bins=20, color='steelblue', edgecolor='white', alpha=0.7)
ax1.axvline(mean_word, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_word:.1f}')
ax1.axvline(median_word, color='green', linestyle='--', linewidth=2, label=f'Median = {median_word:.1f}')
ax1.set_xlabel('Word Count', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Distribution of Word Counts', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Box plot by grade band
ax2 = axes[0, 1]
grade_order = grade_columns
df_stats['Grade_Band'] = pd.Categorical(df_stats['Grade_Band'], 
                                        categories=grade_order, ordered=True)
df_stats_sorted = df_stats.sort_values('Grade_Band')

sns.boxplot(data=df_stats_sorted, x='Grade_Band', y='Word_Count', ax=ax2, 
            palette='RdYlGn_r')
ax2.set_xlabel('Grade Band', fontsize=11)
ax2.set_ylabel('Word Count', fontsize=11)
ax2.set_title('Word Count Distribution by Grade Band', fontsize=12, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Mean word count trend
ax3 = axes[1, 0]
mean_by_grade = df_stats.groupby('Grade_Band')['Word_Count'].mean().reindex(grade_order)
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(grade_order)))[::-1]
ax3.bar(range(len(grade_order)), mean_by_grade.values, color=colors, edgecolor='black')
ax3.set_xticks(range(len(grade_order)))
ax3.set_xticklabels(grade_order, rotation=45, ha='right')
ax3.set_xlabel('Grade Band', fontsize=11)
ax3.set_ylabel('Mean Word Count', fontsize=11)
ax3.set_title('Mean Word Count by Grade Band', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Scatter plot word count vs grade midpoint
ax4 = axes[1, 1]
ax4.scatter(df_stats['Grade_Midpoint'], df_stats['Word_Count'], 
            alpha=0.6, color='steelblue', edgecolors='white', s=50)
ax4.set_xlabel('Grade Midpoint', fontsize=11)
ax4.set_ylabel('Word Count', fontsize=11)
ax4.set_title('Word Count vs Grade Midpoint', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add trend line
z = np.polyfit(df_stats['Grade_Midpoint'], df_stats['Word_Count'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_stats['Grade_Midpoint'].min(), df_stats['Grade_Midpoint'].max(), 100)
ax4.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend Line')
ax4.legend()

plt.tight_layout()
# Ensure output directory exists
output_dir = os.path.join(script_dir, '../outputs/figures')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(script_dir, '../outputs/figures/descriptive_statistics_marking_matrix.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: descriptive_statistics_marking_matrix.png")

# =============================================================================
# STEP 8: KEY FINDINGS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: KEY FINDINGS")
print("=" * 70)

print("\n--- Key Observations ---")
print(f"  1. Average description length: {mean_word:.1f} words")
print(f"  2. Descriptions vary from {word_counts.min()} to {word_counts.max()} words")
print(f"  3. Higher grades tend to have {'more' if z[0] > 0 else 'fewer'} detailed descriptions")
print(f"  4. Distribution is {'positively' if skewness > 0 else 'negatively'} skewed")
print(f"  5. Variability (CV = {CV:.1f}%) indicates {'high' if CV > 50 else 'moderate'} dispersion")

print("\n--- Dataset Reference ---")
print("  Arden University (2024) COM7023 Mathematics for Data Science")
print("  Marking Matrix. Arden University.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF DESCRIPTIVE STATISTICS ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
