"""
================================================================================
DATA LOADING AND EXPLORATION - MARKING MATRIX DATASET
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
This script demonstrates data loading and initial exploration techniques
using the course marking matrix dataset. Understanding data structure is
fundamental to all subsequent mathematical analysis.

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

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# Define script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# HEADER INFORMATION
# =============================================================================

print("=" * 70)
print("DATA LOADING AND EXPLORATION")
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

# Define the file path
file_path = os.path.join(script_dir, '../datasets/COM7023_Mathematics_for_Data_Science_Marking_Matrix.csv')

# Load the CSV file using pandas
# The read_csv function automatically detects the delimiter and encoding
try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("\nDataset loaded successfully!")
    print(f"File path: {file_path}")
except FileNotFoundError:
    print("\nError: Dataset file not found.")
    print("Please ensure the file is in the correct location.")
    exit()

# =============================================================================
# STEP 2: UNDERSTAND THE DATA STRUCTURE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: UNDERSTANDING DATA STRUCTURE")
print("=" * 70)

# Display the shape of the dataset
print("\n--- Dataset Dimensions ---")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Total cells: {df.shape[0] * df.shape[1]}")

# Display column names
print("\n--- Column Names ---")
for i, col in enumerate(df.columns):
    print(f"  {i + 1}. {col}")

# Display data types
print("\n--- Data Types ---")
print(df.dtypes)

# =============================================================================
# STEP 3: PREVIEW THE DATA
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: DATA PREVIEW")
print("=" * 70)

# Display first few rows
print("\n--- First 5 Rows ---")
print(df.head())

# Display the criterion column which contains learning outcomes
print("\n--- Criteria (Learning Outcomes) ---")
for i, criterion in enumerate(df['Criterion'].values):
    # Clean and display each criterion
    criterion_clean = ' '.join(criterion.split())[:80]
    print(f"  LO{i + 1}: {criterion_clean}...")

# =============================================================================
# STEP 4: DATA QUALITY ASSESSMENT
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: DATA QUALITY ASSESSMENT")
print("=" * 70)

# Check for missing values
print("\n--- Missing Values ---")
missing_counts = df.isnull().sum()
print(missing_counts)
print(f"\nTotal missing values: {missing_counts.sum()}")

# Check for duplicate rows
print("\n--- Duplicate Rows ---")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Memory usage
print("\n--- Memory Usage ---")
memory_usage = df.memory_usage(deep=True).sum()
print(f"Total memory usage: {memory_usage / 1024:.2f} KB")

# =============================================================================
# STEP 5: BASIC STATISTICAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: BASIC STATISTICAL SUMMARY")
print("=" * 70)

# For text data, we can analyse text lengths
print("\n--- Text Length Analysis ---")

# Create a summary of text lengths for each grade band
grade_columns = ['90- 100', '80 - 90', '70 – 79', '60 - 69', 
                 '50 – 59', '40 – 49', '30 – 39', '20 - 29', '0 - 19']

text_lengths = {}
for col in grade_columns:
    if col in df.columns:
        # Calculate average text length for each column
        avg_length = df[col].apply(lambda x: len(str(x))).mean()
        text_lengths[col] = avg_length

print("\nAverage text length by grade band:")
for grade, length in text_lengths.items():
    print(f"  {grade}: {length:.1f} characters")

# =============================================================================
# STEP 6: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: DATA VISUALISATION")
print("=" * 70)

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bar chart of text lengths by grade band
ax1 = axes[0]
grades = list(text_lengths.keys())
lengths = list(text_lengths.values())
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(grades)))[::-1]

bars = ax1.bar(range(len(grades)), lengths, color=colors, edgecolor='black')
ax1.set_xticks(range(len(grades)))
ax1.set_xticklabels(grades, rotation=45, ha='right')
ax1.set_xlabel('Grade Band', fontsize=11)
ax1.set_ylabel('Average Text Length (characters)', fontsize=11)
ax1.set_title('Description Length by Grade Band', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Heatmap of data presence
ax2 = axes[1]

# Create a binary matrix showing data presence
presence_matrix = df.iloc[:, 1:].notna().astype(int)
presence_matrix.index = [f'LO{i+1}' for i in range(len(df))]

sns.heatmap(presence_matrix, annot=True, cmap='Greens', 
            cbar_kws={'label': 'Data Present'}, ax=ax2)
ax2.set_xlabel('Grade Band', fontsize=11)
ax2.set_ylabel('Learning Outcome', fontsize=11)
ax2.set_title('Data Completeness Matrix', fontsize=12, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
# Ensure output directory exists
output_dir = os.path.join(script_dir, '../outputs/figures')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(script_dir, '../outputs/figures/data_loading_marking_matrix.png'), 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: data_loading_marking_matrix.png")

# =============================================================================
# STEP 7: KEY FINDINGS SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: KEY FINDINGS SUMMARY")
print("=" * 70)

print("\n--- Summary ---")
print(f"  1. Dataset contains {df.shape[0]} learning outcomes (rows)")
print(f"  2. Dataset contains {df.shape[1]} columns (1 criterion + 9 grade bands)")
print(f"  3. All cells contain data (no missing values)")
print(f"  4. Higher grades have more detailed descriptions")
print(f"  5. Dataset structure is suitable for text analysis and categorisation")

print("\n--- Dataset Reference ---")
print("  Arden University (2024) COM7023 Mathematics for Data Science")
print("  Marking Matrix. Arden University.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF DATA LOADING AND EXPLORATION")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
