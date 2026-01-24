"""
================================================================================
DATA LOADING AND EXPLORATION - TRIPLET BIRTHS DATASET
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
using the Human Multiple Births Database. This real-world demographic dataset
contains information about triplet births in France across multiple years.

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

# Set display options for better readability
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# HEADER INFORMATION
# =============================================================================

print("=" * 70)
print("DATA LOADING AND EXPLORATION")
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
# STEP 1: LOAD THE DATASET
# =============================================================================

print("\n" + "=" * 70)
print("STEP 1: LOADING THE DATASET")
print("=" * 70)

# Define the file path
file_path = '../datasets/FRA_InputData_25.11.2024.xlsx'

# Load the Excel file using pandas
# We specify the sheet name as 'input data' based on the dataset structure
try:
    df = pd.read_excel(file_path, sheet_name='input data')
    print("\nDataset loaded successfully!")
    print(f"File path: {file_path}")
    print(f"Sheet name: input data")
except FileNotFoundError:
    print("\nError: Dataset file not found.")
    print("Please download from: https://www.twinbirths.org/en/data-metadata/")
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
print("\n--- First 10 Rows ---")
print(df.head(10))

# Display last few rows
print("\n--- Last 5 Rows ---")
print(df.tail())

# =============================================================================
# STEP 4: DATA QUALITY ASSESSMENT
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: DATA QUALITY ASSESSMENT")
print("=" * 70)

# Check for missing values
print("\n--- Missing Values per Column ---")
missing_counts = df.isnull().sum()
missing_percent = (df.isnull().sum() / len(df)) * 100

for col in df.columns:
    if missing_counts[col] > 0:
        print(f"  {col}: {missing_counts[col]} ({missing_percent[col]:.1f}%)")

print(f"\nTotal missing values: {missing_counts.sum()}")

# Check for duplicate rows
print("\n--- Duplicate Rows ---")
duplicates = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

# Check unique years
print("\n--- Year Range ---")
if 'Year' in df.columns:
    df_clean = df.dropna(subset=['Year'])
    years = df_clean['Year'].astype(int)
    print(f"Earliest year: {years.min()}")
    print(f"Latest year: {years.max()}")
    print(f"Number of years with data: {len(years.unique())}")

# =============================================================================
# STEP 5: BASIC STATISTICAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: BASIC STATISTICAL SUMMARY")
print("=" * 70)

# Get numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

print("\n--- Numeric Columns Summary ---")
print(df[numeric_cols].describe())

# Focus on triplet deliveries
print("\n--- Triplet Deliveries Statistics ---")
if 'Triplet_deliveries' in df.columns:
    triplet_col = 'Triplet_deliveries'
    triplet_data = df[triplet_col].dropna()
    
    print(f"  Count: {len(triplet_data)}")
    print(f"  Mean: {triplet_data.mean():.2f}")
    print(f"  Median: {triplet_data.median():.2f}")
    print(f"  Standard Deviation: {triplet_data.std():.2f}")
    print(f"  Minimum: {triplet_data.min():.0f}")
    print(f"  Maximum: {triplet_data.max():.0f}")
    print(f"  Range: {triplet_data.max() - triplet_data.min():.0f}")

# =============================================================================
# STEP 6: DATA CLEANING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: DATA CLEANING")
print("=" * 70)

# Remove rows with missing values in key columns
df_cleaned = df.dropna(subset=['Year', 'Triplet_deliveries'])

# Convert year to integer
df_cleaned['Year'] = df_cleaned['Year'].astype(int)

# Filter to a consistent time period (1990-2020)
start_year = 1990
end_year = 2020
df_filtered = df_cleaned[(df_cleaned['Year'] >= start_year) & 
                          (df_cleaned['Year'] <= end_year)].copy()

print(f"\n--- Data Cleaning Results ---")
print(f"  Original rows: {len(df)}")
print(f"  After removing missing values: {len(df_cleaned)}")
print(f"  After filtering years ({start_year}-{end_year}): {len(df_filtered)}")

# =============================================================================
# STEP 7: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: DATA VISUALISATION")
print("=" * 70)

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Time series of triplet births
ax1 = axes[0, 0]
ax1.plot(df_filtered['Year'], df_filtered['Triplet_deliveries'], 
         marker='o', linewidth=2, markersize=5, color='steelblue')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Number of Triplet Deliveries', fontsize=11)
ax1.set_title('Triplet Births in France (1990-2020)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Histogram of triplet births
ax2 = axes[0, 1]
ax2.hist(df_filtered['Triplet_deliveries'], bins=15, color='steelblue', 
         edgecolor='white', alpha=0.7)
ax2.axvline(df_filtered['Triplet_deliveries'].mean(), color='red', 
            linestyle='--', linewidth=2, label='Mean')
ax2.axvline(df_filtered['Triplet_deliveries'].median(), color='green', 
            linestyle='--', linewidth=2, label='Median')
ax2.set_xlabel('Number of Triplet Deliveries', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Distribution of Annual Triplet Births', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Box plot
ax3 = axes[1, 0]
ax3.boxplot(df_filtered['Triplet_deliveries'], vert=True)
ax3.set_ylabel('Number of Triplet Deliveries', fontsize=11)
ax3.set_title('Box Plot of Triplet Births', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Missing values heatmap (for original data)
ax4 = axes[1, 1]
missing_data = df.isnull()
# Select first 20 rows and key columns for visualisation
cols_to_show = ['Year', 'Total_births', 'Twin_deliveries', 'Triplet_deliveries']
cols_available = [c for c in cols_to_show if c in df.columns]
missing_subset = missing_data[cols_available].head(50)

sns.heatmap(missing_subset, cbar=True, cmap='YlOrRd', ax=ax4,
            cbar_kws={'label': 'Missing'})
ax4.set_xlabel('Variables', fontsize=11)
ax4.set_ylabel('Rows', fontsize=11)
ax4.set_title('Missing Data Pattern (First 50 Rows)', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('../outputs/figures/data_loading_triplet_births.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: data_loading_triplet_births.png")

# =============================================================================
# STEP 8: KEY FINDINGS SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: KEY FINDINGS SUMMARY")
print("=" * 70)

print("\n--- Summary ---")
print(f"  1. Dataset spans from {df_cleaned['Year'].min()} to {df_cleaned['Year'].max()}")
print(f"  2. Analysis period filtered to {start_year}-{end_year} ({len(df_filtered)} years)")
print(f"  3. Average triplet births: {df_filtered['Triplet_deliveries'].mean():.1f} per year")
print(f"  4. Data shows temporal patterns suitable for time series analysis")
print(f"  5. Dataset is suitable for Poisson distribution modelling")

print("\n--- Dataset Reference ---")
print("  Human Multiple Births Database (2024) FRA_InputData_25.11.2024.xlsx.")
print("  Available at: https://www.twinbirths.org/en/data-metadata/")
print("  (Accessed: 25 November 2024).")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF DATA LOADING AND EXPLORATION")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
