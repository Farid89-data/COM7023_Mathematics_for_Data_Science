"""
================================================================================
PROBABILITY DISTRIBUTIONS - MARKING MATRIX DATASET
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
This script demonstrates probability concepts using the marking matrix dataset.
We explore discrete probability distributions and conditional probability
applied to grade distributions and assessment criteria.

MATHEMATICAL CONCEPTS:
- Discrete probability distributions
- Probability mass functions
- Conditional probability
- Bayes' theorem
- Expected value and variance

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
from math import factorial

# =============================================================================
# CONFIGURATION
# =============================================================================

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# HEADER INFORMATION
# =============================================================================

print("=" * 70)
print("PROBABILITY DISTRIBUTIONS")
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

file_path = '../datasets/COM7023_Mathematics_for_Data_Science_Marking_Matrix.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("\nDataset loaded successfully!")
except FileNotFoundError:
    print("\nError: Dataset file not found.")
    exit()

# Define grade bands and midpoints
grade_columns = ['90- 100', '80 - 90', '70 – 79', '60 - 69', 
                 '50 – 59', '40 – 49', '30 – 39', '20 - 29', '0 - 19']

grade_midpoints = {
    '90- 100': 95, '80 - 90': 85, '70 – 79': 74.5, '60 - 69': 64.5,
    '50 – 59': 54.5, '40 – 49': 44.5, '30 – 39': 34.5, '20 - 29': 24.5, '0 - 19': 9.5
}

print(f"\nNumber of Learning Outcomes: {len(df)}")
print(f"Number of Grade Bands: {len(grade_columns)}")

# =============================================================================
# STEP 2: DISCRETE PROBABILITY FUNDAMENTALS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: DISCRETE PROBABILITY FUNDAMENTALS")
print("=" * 70)

print("\n--- Probability Axioms ---")
print("\n  1. P(A) ≥ 0 for any event A")
print("  2. P(S) = 1 where S is the sample space")
print("  3. P(A ∪ B) = P(A) + P(B) if A and B are mutually exclusive")

print("\n--- Creating a Hypothetical Grade Distribution ---")
print("\n  Let's model a hypothetical grade distribution for this module.")

# Create a simulated grade distribution (based on typical academic patterns)
# This represents P(Grade = g) for each grade band
grade_probs = {
    '90- 100': 0.05,   # 5% get first class distinction
    '80 - 90': 0.10,   # 10% get first class
    '70 – 79': 0.20,   # 20% get upper second
    '60 - 69': 0.25,   # 25% get lower second
    '50 – 59': 0.20,   # 20% get third
    '40 – 49': 0.12,   # 12% pass
    '30 – 39': 0.05,   # 5% marginal fail
    '20 - 29': 0.02,   # 2% fail
    '0 - 19': 0.01     # 1% fail
}

print("\n  Hypothetical Grade Distribution P(G = g):")
print("\n  Grade Band    Probability    P(G = g)")
print("  " + "-" * 45)
for grade, prob in grade_probs.items():
    bar = "█" * int(prob * 50)
    print(f"  {grade:<12}    {prob:.2f}         {bar}")

# Verify probabilities sum to 1
total_prob = sum(grade_probs.values())
print(f"\n  Verification: ΣP(G = g) = {total_prob:.2f} (should be 1.00)")

# =============================================================================
# STEP 3: EXPECTED VALUE AND VARIANCE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: EXPECTED VALUE AND VARIANCE")
print("=" * 70)

print("\n--- Expected Value (Mean) ---")
print("\n  Formula: E[X] = Σ xᵢ × P(X = xᵢ)")
print("\n  For discrete random variable X:")

# Calculate expected grade
expected_grade = sum(grade_midpoints[g] * grade_probs[g] for g in grade_columns)

print(f"\n  Step-by-step calculation:")
print(f"\n  E[Grade] = Σ (midpoint × probability)")
print(f"\n  Grade Band    Midpoint    P(G=g)    Contribution")
print("  " + "-" * 55)
for g in grade_columns:
    contribution = grade_midpoints[g] * grade_probs[g]
    print(f"  {g:<12}  {grade_midpoints[g]:>6.1f}     {grade_probs[g]:.2f}     {contribution:>8.2f}")
print("  " + "-" * 55)
print(f"  Expected Grade (Mean): E[G] = {expected_grade:.2f}")

print("\n--- Variance ---")
print("\n  Formula: Var(X) = E[X²] - (E[X])²")
print("          = Σ xᵢ² × P(X = xᵢ) - μ²")

# Calculate E[X²]
e_x_squared = sum((grade_midpoints[g] ** 2) * grade_probs[g] for g in grade_columns)
variance = e_x_squared - expected_grade ** 2
std_dev = np.sqrt(variance)

print(f"\n  Calculation:")
print(f"  E[G²] = Σ(midpoint² × probability) = {e_x_squared:.2f}")
print(f"  Var(G) = E[G²] - (E[G])² = {e_x_squared:.2f} - ({expected_grade:.2f})²")
print(f"  Var(G) = {e_x_squared:.2f} - {expected_grade**2:.2f} = {variance:.2f}")
print(f"  SD(G) = √Var(G) = √{variance:.2f} = {std_dev:.2f}")

# =============================================================================
# STEP 4: CONDITIONAL PROBABILITY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: CONDITIONAL PROBABILITY")
print("=" * 70)

print("\n--- Conditional Probability Formula ---")
print("\n  P(A|B) = P(A ∩ B) / P(B)")
print("\n  Reads: Probability of A given B")

# Define events
# Event A: Student passes (grade ≥ 40)
# Event B: Student achieves distinction (grade ≥ 70)

P_pass = sum(grade_probs[g] for g in grade_columns if grade_midpoints[g] >= 44.5)
P_distinction = sum(grade_probs[g] for g in grade_columns if grade_midpoints[g] >= 74.5)
P_fail = sum(grade_probs[g] for g in grade_columns if grade_midpoints[g] < 44.5)

print(f"\n  Event Probabilities:")
print(f"  P(Pass) = P(Grade ≥ 40) = {P_pass:.2f}")
print(f"  P(Distinction) = P(Grade ≥ 70) = {P_distinction:.2f}")
print(f"  P(Fail) = P(Grade < 40) = {P_fail:.2f}")

# Conditional: P(Distinction | Pass)
# Since distinction implies pass, P(Distinction ∩ Pass) = P(Distinction)
P_distinction_given_pass = P_distinction / P_pass

print(f"\n  Conditional Probability:")
print(f"  P(Distinction | Pass) = P(Distinction ∩ Pass) / P(Pass)")
print(f"                        = P(Distinction) / P(Pass)")
print(f"                        = {P_distinction:.2f} / {P_pass:.2f}")
print(f"                        = {P_distinction_given_pass:.4f}")
print(f"\n  Interpretation: Given a student passes, there is a")
print(f"  {P_distinction_given_pass*100:.1f}% chance they achieve distinction.")

# =============================================================================
# STEP 5: CUMULATIVE DISTRIBUTION FUNCTION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: CUMULATIVE DISTRIBUTION FUNCTION (CDF)")
print("=" * 70)

print("\n--- CDF Definition ---")
print("\n  F(x) = P(X ≤ x) = Σ P(X = xᵢ) for all xᵢ ≤ x")

# Calculate CDF
cdf = {}
cumsum = 0
print(f"\n  Grade Band    P(G = g)    F(g) = P(G ≤ g)")
print("  " + "-" * 50)
for g in grade_columns[::-1]:  # Start from lowest grade
    cumsum += grade_probs[g]
    cdf[g] = cumsum
    print(f"  {g:<12}    {grade_probs[g]:.2f}          {cumsum:.2f}")

# Use CDF to calculate probabilities
print(f"\n--- Using CDF for Probability Calculations ---")
print(f"\n  P(50 ≤ G ≤ 69) = F(69) - F(49)")
F_69 = sum(grade_probs[g] for g in grade_columns if grade_midpoints[g] <= 69)
F_49 = sum(grade_probs[g] for g in grade_columns if grade_midpoints[g] < 50)
P_50_to_69 = F_69 - F_49
print(f"                 = {F_69:.2f} - {F_49:.2f}")
print(f"                 = {P_50_to_69:.2f}")

# =============================================================================
# STEP 6: BINOMIAL DISTRIBUTION EXAMPLE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: BINOMIAL DISTRIBUTION EXAMPLE")
print("=" * 70)

print("\n--- Binomial Distribution ---")
print("\n  P(X = k) = C(n,k) × p^k × (1-p)^(n-k)")
print("\n  where C(n,k) = n! / (k! × (n-k)!)")
print("\n  Used when:")
print("  - Fixed number of trials (n)")
print("  - Two possible outcomes (success/failure)")
print("  - Constant probability of success (p)")
print("  - Independent trials")

def binomial_pmf(n, k, p):
    """Calculate binomial probability P(X = k)"""
    C_nk = factorial(n) / (factorial(k) * factorial(n - k))
    return C_nk * (p ** k) * ((1 - p) ** (n - k))

# Example: Probability of exactly k students passing out of n
n = 10  # Number of students
p = P_pass  # Probability of passing

print(f"\n  Example: Class of {n} students, P(pass) = {p:.2f}")
print(f"\n  What is P(exactly k students pass)?")
print(f"\n  k    P(X = k)      Calculation")
print("  " + "-" * 60)
for k in range(n + 1):
    prob = binomial_pmf(n, k, p)
    calc = f"C({n},{k}) × {p:.2f}^{k} × {1-p:.2f}^{n-k}"
    print(f"  {k:<3}  {prob:.6f}      {calc}")

# Expected value for binomial
E_binomial = n * p
Var_binomial = n * p * (1 - p)
print(f"\n  Expected number of passes: E[X] = n × p = {n} × {p:.2f} = {E_binomial:.2f}")
print(f"  Variance: Var(X) = n × p × (1-p) = {Var_binomial:.2f}")

# =============================================================================
# STEP 7: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Grade Distribution (PMF)
ax1 = axes[0, 0]
grades = list(grade_probs.keys())
probs = list(grade_probs.values())
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(grades)))[::-1]
bars = ax1.bar(range(len(grades)), probs, color=colors, edgecolor='black')
ax1.set_xticks(range(len(grades)))
ax1.set_xticklabels(grades, rotation=45, ha='right')
ax1.set_xlabel('Grade Band', fontsize=11)
ax1.set_ylabel('Probability P(G = g)', fontsize=11)
ax1.set_title('Probability Mass Function (PMF)\nGrade Distribution', 
              fontsize=12, fontweight='bold')
ax1.axvline(x=list(grade_midpoints.values()).index(expected_grade) if expected_grade in 
            grade_midpoints.values() else 3, color='red', linestyle='--', 
            label=f'E[G] ≈ {expected_grade:.0f}')
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: CDF
ax2 = axes[0, 1]
cdf_values = []
cumsum = 0
for g in grade_columns[::-1]:
    cumsum += grade_probs[g]
    cdf_values.append(cumsum)
cdf_values = cdf_values[::-1]

ax2.step(range(len(grades)), cdf_values, where='mid', color='steelblue', linewidth=2)
ax2.scatter(range(len(grades)), cdf_values, color='steelblue', s=50, zorder=5)
ax2.set_xticks(range(len(grades)))
ax2.set_xticklabels(grades, rotation=45, ha='right')
ax2.set_xlabel('Grade Band', fontsize=11)
ax2.set_ylabel('Cumulative Probability F(g)', fontsize=11)
ax2.set_title('Cumulative Distribution Function (CDF)', fontsize=12, fontweight='bold')
ax2.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Median')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Binomial Distribution
ax3 = axes[1, 0]
k_values = range(n + 1)
binomial_probs = [binomial_pmf(n, k, p) for k in k_values]
ax3.bar(k_values, binomial_probs, color='steelblue', edgecolor='white', alpha=0.8)
ax3.axvline(E_binomial, color='red', linestyle='--', linewidth=2, 
            label=f'E[X] = {E_binomial:.1f}')
ax3.set_xlabel('Number of Students Passing (k)', fontsize=11)
ax3.set_ylabel('Probability P(X = k)', fontsize=11)
ax3.set_title(f'Binomial Distribution: n={n}, p={p:.2f}', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Expected Value illustration
ax4 = axes[1, 1]
midpoints = [grade_midpoints[g] for g in grade_columns]
contributions = [grade_midpoints[g] * grade_probs[g] for g in grade_columns]
ax4.bar(range(len(grades)), contributions, color='lightgreen', 
        edgecolor='black', label='Contribution to E[G]')
ax4.set_xticks(range(len(grades)))
ax4.set_xticklabels(grades, rotation=45, ha='right')
ax4.set_xlabel('Grade Band', fontsize=11)
ax4.set_ylabel('Contribution (midpoint × probability)', fontsize=11)
ax4.set_title(f'Expected Value Calculation\nE[G] = Σ(x × P(x)) = {expected_grade:.1f}', 
              fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('../outputs/figures/probability_distributions_marking_matrix.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: probability_distributions_marking_matrix.png")

# =============================================================================
# STEP 8: KEY FINDINGS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: KEY FINDINGS")
print("=" * 70)

print("\n--- Summary Statistics ---")
print(f"  Expected Grade: {expected_grade:.2f}")
print(f"  Standard Deviation: {std_dev:.2f}")
print(f"  Probability of Passing: {P_pass:.2%}")
print(f"  Probability of Distinction: {P_distinction:.2%}")

print("\n--- Key Concepts Demonstrated ---")
print("  1. Probability Mass Function (PMF) for discrete distributions")
print("  2. Expected value as weighted average")
print("  3. Variance as measure of spread")
print("  4. Conditional probability using Bayes' theorem")
print("  5. Cumulative Distribution Function (CDF)")
print("  6. Binomial distribution for binary outcomes")

print("\n--- Dataset Reference ---")
print("  Arden University (2024) COM7023 Mathematics for Data Science")
print("  Marking Matrix. Arden University.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF PROBABILITY DISTRIBUTIONS ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
