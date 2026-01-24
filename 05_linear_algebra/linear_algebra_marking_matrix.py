"""
================================================================================
LINEAR ALGEBRA - MARKING MATRIX DATASET
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
This script demonstrates linear algebra concepts using the marking matrix
dataset. We construct matrices from the data and perform fundamental matrix
operations essential for data science applications.

MATHEMATICAL CONCEPTS:
- Vectors and vector operations
- Matrix representation and dimensions
- Matrix arithmetic (addition, multiplication)
- Transpose and determinant
- Matrix inverse
- Eigenvalues and eigenvectors

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
np.set_printoptions(precision=4, suppress=True)

# =============================================================================
# HEADER INFORMATION
# =============================================================================

print("=" * 70)
print("LINEAR ALGEBRA")
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

# Define grade columns and midpoints
grade_columns = ['90- 100', '80 - 90', '70 – 79', '60 - 69', 
                 '50 – 59', '40 – 49', '30 – 39', '20 - 29', '0 - 19']

grade_midpoints = {
    '90- 100': 95, '80 - 90': 85, '70 – 79': 74.5, '60 - 69': 64.5,
    '50 – 59': 54.5, '40 – 49': 44.5, '30 – 39': 34.5, '20 - 29': 24.5, '0 - 19': 9.5
}

# Create numerical features matrix
text_features = []
for idx, row in df.iterrows():
    row_features = []
    for grade_col in grade_columns:
        if grade_col in df.columns:
            text = str(row[grade_col])
            word_count = len(text.split())
            row_features.append(word_count)
    text_features.append(row_features)

# Create the feature matrix
feature_matrix = np.array(text_features)
print(f"\nFeature matrix shape: {feature_matrix.shape}")
print(f"  - Rows (Learning Outcomes): {feature_matrix.shape[0]}")
print(f"  - Columns (Grade Bands): {feature_matrix.shape[1]}")

# =============================================================================
# STEP 2: VECTORS - FUNDAMENTAL CONCEPTS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: VECTORS - FUNDAMENTAL CONCEPTS")
print("=" * 70)

print("\n--- What is a Vector? ---")
print("\n  A vector is an ordered list of numbers, representing a point")
print("  or direction in n-dimensional space.")
print("\n       ┌ v₁ ┐")
print("  v =  │ v₂ │   ∈ ℝⁿ")
print("       │ ⋮  │")
print("       └ vₙ ┘")

# Extract vectors from our data
# Each row is a vector representing word counts across grade bands
vector_LO1 = feature_matrix[0, :]
vector_LO2 = feature_matrix[1, :]

print("\n--- Vectors from Dataset ---")
print(f"\n  Vector for LO1 (word counts across grades):")
print(f"  v₁ = {vector_LO1}")
print(f"\n  Vector for LO2 (word counts across grades):")
print(f"  v₂ = {vector_LO2}")

# Vector dimensions
print(f"\n  Dimension of vectors: n = {len(vector_LO1)}")

# =============================================================================
# STEP 3: VECTOR OPERATIONS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: VECTOR OPERATIONS")
print("=" * 70)

print("\n" + "-" * 50)
print("3.1 VECTOR ADDITION")
print("-" * 50)

print("\n  Formula: u + v = [u₁ + v₁, u₂ + v₂, ..., uₙ + vₙ]")

# Vector addition
vector_sum = vector_LO1 + vector_LO2

print(f"\n  v₁ = {vector_LO1}")
print(f"  v₂ = {vector_LO2}")
print(f"\n  v₁ + v₂ = {vector_sum}")

print("\n  Step-by-step:")
for i in range(min(3, len(vector_LO1))):
    print(f"    Position {i+1}: {vector_LO1[i]} + {vector_LO2[i]} = {vector_sum[i]}")
print("    ...")

print("\n" + "-" * 50)
print("3.2 SCALAR MULTIPLICATION")
print("-" * 50)

print("\n  Formula: c × v = [c×v₁, c×v₂, ..., c×vₙ]")

scalar = 2
scaled_vector = scalar * vector_LO1

print(f"\n  c = {scalar}")
print(f"  v₁ = {vector_LO1}")
print(f"\n  c × v₁ = {scalar} × v₁ = {scaled_vector}")

print("\n" + "-" * 50)
print("3.3 DOT PRODUCT (INNER PRODUCT)")
print("-" * 50)

print("\n  Formula: u · v = Σᵢ uᵢ × vᵢ = u₁v₁ + u₂v₂ + ... + uₙvₙ")
print("\n  The dot product measures similarity between vectors.")

# Calculate dot product
dot_product = np.dot(vector_LO1, vector_LO2)

print(f"\n  v₁ = {vector_LO1}")
print(f"  v₂ = {vector_LO2}")
print(f"\n  Step-by-step calculation:")
print(f"  v₁ · v₂ = ", end="")

products = vector_LO1 * vector_LO2
for i, p in enumerate(products):
    if i < len(products) - 1:
        print(f"{vector_LO1[i]}×{vector_LO2[i]} + ", end="")
    else:
        print(f"{vector_LO1[i]}×{vector_LO2[i]}")

print(f"         = ", end="")
for i, p in enumerate(products):
    if i < len(products) - 1:
        print(f"{p} + ", end="")
    else:
        print(f"{p}")

print(f"         = {dot_product}")

print("\n" + "-" * 50)
print("3.4 VECTOR MAGNITUDE (NORM)")
print("-" * 50)

print("\n  Formula: ||v|| = √(Σᵢ vᵢ²) = √(v₁² + v₂² + ... + vₙ²)")
print("\n  The Euclidean norm measures the 'length' of a vector.")

# Calculate magnitude
magnitude_LO1 = np.linalg.norm(vector_LO1)
magnitude_LO2 = np.linalg.norm(vector_LO2)

print(f"\n  ||v₁|| = √({' + '.join([f'{x}²' for x in vector_LO1[:3]])} + ...)")
print(f"        = √({sum(vector_LO1**2)})")
print(f"        = {magnitude_LO1:.4f}")

print(f"\n  ||v₂|| = {magnitude_LO2:.4f}")

print("\n" + "-" * 50)
print("3.5 COSINE SIMILARITY")
print("-" * 50)

print("\n  Formula: cos(θ) = (u · v) / (||u|| × ||v||)")
print("\n  Measures the angle between two vectors:")
print("  - cos(θ) = 1: vectors point same direction")
print("  - cos(θ) = 0: vectors are orthogonal")
print("  - cos(θ) = -1: vectors point opposite directions")

# Calculate cosine similarity
cosine_sim = dot_product / (magnitude_LO1 * magnitude_LO2)
angle_radians = np.arccos(np.clip(cosine_sim, -1, 1))
angle_degrees = np.degrees(angle_radians)

print(f"\n  cos(θ) = (v₁ · v₂) / (||v₁|| × ||v₂||)")
print(f"         = {dot_product} / ({magnitude_LO1:.4f} × {magnitude_LO2:.4f})")
print(f"         = {dot_product} / {magnitude_LO1 * magnitude_LO2:.4f}")
print(f"         = {cosine_sim:.6f}")

print(f"\n  θ = arccos({cosine_sim:.6f})")
print(f"    = {angle_radians:.4f} radians")
print(f"    = {angle_degrees:.2f}°")

print(f"\n  Interpretation: LO1 and LO2 are {'highly' if cosine_sim > 0.9 else 'moderately' if cosine_sim > 0.7 else 'weakly'} similar")

# =============================================================================
# STEP 4: MATRICES - FUNDAMENTAL CONCEPTS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: MATRICES - FUNDAMENTAL CONCEPTS")
print("=" * 70)

print("\n--- What is a Matrix? ---")
print("\n  A matrix is a rectangular array of numbers with m rows and n columns:")
print("\n       ┌ a₁₁  a₁₂  ...  a₁ₙ ┐")
print("  A =  │ a₂₁  a₂₂  ...  a₂ₙ │   ∈ ℝᵐˣⁿ")
print("       │  ⋮    ⋮   ⋱    ⋮  │")
print("       └ aₘ₁  aₘ₂  ...  aₘₙ ┘")

# Display our feature matrix
A = feature_matrix
print(f"\n--- Feature Matrix A (Word Counts) ---")
print(f"\n  Dimensions: {A.shape[0]} × {A.shape[1]} (m × n)")
print(f"\n  A = ")
for i, row in enumerate(A):
    print(f"      LO{i+1}: {row}")

# Matrix notation
print(f"\n  Notation:")
print(f"  - A ∈ ℝ^({A.shape[0]}×{A.shape[1]})")
print(f"  - Element a_ij represents word count for LO_i at grade band j")
print(f"  - Example: a₁₂ = A[0,1] = {A[0,1]} (LO1, grade band 80-90)")

# =============================================================================
# STEP 5: MATRIX OPERATIONS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: MATRIX OPERATIONS")
print("=" * 70)

print("\n" + "-" * 50)
print("5.1 MATRIX TRANSPOSE")
print("-" * 50)

print("\n  Definition: (Aᵀ)ᵢⱼ = aⱼᵢ")
print("\n  Transpose swaps rows and columns:")
print("  - If A is m × n, then Aᵀ is n × m")

A_transpose = A.T

print(f"\n  Original A ({A.shape[0]}×{A.shape[1]}):")
for row in A:
    print(f"    {row}")

print(f"\n  Transpose Aᵀ ({A_transpose.shape[0]}×{A_transpose.shape[1]}):")
for row in A_transpose:
    print(f"    {row}")

print("\n  Properties of Transpose:")
print("  - (Aᵀ)ᵀ = A")
print("  - (A + B)ᵀ = Aᵀ + Bᵀ")
print("  - (AB)ᵀ = BᵀAᵀ")
print("  - (cA)ᵀ = cAᵀ")

print("\n" + "-" * 50)
print("5.2 MATRIX-VECTOR MULTIPLICATION")
print("-" * 50)

print("\n  Formula: y = Ax where A ∈ ℝᵐˣⁿ, x ∈ ℝⁿ, y ∈ ℝᵐ")
print("\n  yᵢ = Σⱼ aᵢⱼxⱼ (dot product of row i with vector x)")

# Create a weight vector for grade bands
x = np.array([grade_midpoints[g] for g in grade_columns])
x_normalized = x / x.sum()  # Normalize weights

print(f"\n  Weight vector x (normalized grade midpoints):")
print(f"  x = {x_normalized}")

# Matrix-vector multiplication
y = A @ x_normalized

print(f"\n  Result y = Ax:")
for i, val in enumerate(y):
    print(f"    y{i+1} (LO{i+1} weighted score) = {val:.4f}")

print("\n" + "-" * 50)
print("5.3 MATRIX MULTIPLICATION")
print("-" * 50)

print("\n  Formula: C = AB where A ∈ ℝᵐˣⁿ, B ∈ ℝⁿˣᵖ, C ∈ ℝᵐˣᵖ")
print("\n  cᵢⱼ = Σₖ aᵢₖbₖⱼ")
print("\n  Requirement: Number of columns in A = Number of rows in B")

# Compute AᵀA (Gram matrix)
ATA = A.T @ A

print(f"\n  Computing AᵀA (Gram Matrix):")
print(f"  A ∈ ℝ^{A.shape}, Aᵀ ∈ ℝ^{A.T.shape}")
print(f"  AᵀA ∈ ℝ^{ATA.shape}")

print(f"\n  AᵀA = ")
for row in ATA[:5, :5]:  # Show first 5x5
    print(f"    {row}")
if ATA.shape[0] > 5:
    print("    ...")

print("\n  Properties of AᵀA:")
print("  - Always symmetric: (AᵀA)ᵀ = AᵀA")
print("  - Always positive semi-definite")
print("  - Diagonal elements are sum of squared column values")

# Compute AAᵀ
AAT = A @ A.T

print(f"\n  Computing AAᵀ:")
print(f"  AAᵀ ∈ ℝ^{AAT.shape}")
print(f"\n  AAᵀ = ")
for row in AAT:
    print(f"    {row}")

# =============================================================================
# STEP 6: SQUARE MATRIX OPERATIONS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: SQUARE MATRIX OPERATIONS")
print("=" * 70)

# Use AAᵀ as our square matrix (4×4)
S = AAT
print(f"\n  Using S = AAᵀ as our square matrix ({S.shape[0]}×{S.shape[1]})")

print("\n" + "-" * 50)
print("6.1 TRACE")
print("-" * 50)

print("\n  Definition: tr(A) = Σᵢ aᵢᵢ (sum of diagonal elements)")

trace_S = np.trace(S)
diagonal = np.diag(S)

print(f"\n  Diagonal elements: {diagonal}")
print(f"  tr(S) = {' + '.join([str(int(d)) for d in diagonal])}")
print(f"        = {trace_S:.0f}")

print("\n" + "-" * 50)
print("6.2 DETERMINANT")
print("-" * 50)

print("\n  The determinant measures how a matrix scales volume.")
print("\n  For 2×2 matrix:")
print("       |a  b|")
print("  det  |c  d| = ad - bc")

print("\n  For larger matrices, use cofactor expansion or LU decomposition.")

det_S = np.linalg.det(S)

print(f"\n  det(S) = {det_S:.4f}")

if det_S != 0:
    print(f"\n  Since det(S) ≠ 0, matrix S is invertible")
else:
    print(f"\n  Since det(S) = 0, matrix S is singular (not invertible)")

print("\n" + "-" * 50)
print("6.3 MATRIX INVERSE")
print("-" * 50)

print("\n  Definition: A⁻¹ is the matrix such that AA⁻¹ = A⁻¹A = I")
print("\n  Exists only if det(A) ≠ 0")

if det_S != 0:
    S_inv = np.linalg.inv(S)
    
    print(f"\n  S⁻¹ = ")
    for row in S_inv:
        print(f"    [{', '.join([f'{x:.6f}' for x in row])}]")
    
    # Verify inverse
    identity_check = S @ S_inv
    print(f"\n  Verification: S × S⁻¹ ≈ I")
    print(f"  (S × S⁻¹) = ")
    for row in identity_check:
        print(f"    [{', '.join([f'{x:.4f}' for x in row])}]")
else:
    print("\n  Matrix is singular, inverse does not exist.")

# =============================================================================
# STEP 7: EIGENVALUES AND EIGENVECTORS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: EIGENVALUES AND EIGENVECTORS")
print("=" * 70)

print("\n--- Fundamental Definition ---")
print("\n  Eigenvalue equation: Av = λv")
print("\n  For a square matrix A:")
print("  - λ (lambda) is an eigenvalue")
print("  - v is the corresponding eigenvector")
print("  - The eigenvector direction is preserved under transformation")

print("\n--- Finding Eigenvalues ---")
print("\n  Solve: det(A - λI) = 0")
print("  This is the characteristic polynomial")

# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(S)

print(f"\n--- Eigenanalysis of S ({S.shape[0]}×{S.shape[1]}) ---")

print(f"\n  Eigenvalues (λ):")
for i, ev in enumerate(eigenvalues):
    print(f"    λ{i+1} = {ev:.4f}")

print(f"\n  Eigenvectors (columns of V):")
for i in range(len(eigenvalues)):
    print(f"    v{i+1} = {eigenvectors[:, i]}")

# Verify eigenvalue equation
print(f"\n--- Verification: Av = λv ---")
for i in range(min(2, len(eigenvalues))):  # Verify first 2
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    
    Av = S @ v
    lambda_v = lam * v
    
    print(f"\n  For λ{i+1} = {lam:.4f}:")
    print(f"    Av      = {Av}")
    print(f"    λv      = {lambda_v}")
    print(f"    Match?  = {np.allclose(Av, lambda_v)}")

print("\n--- Properties of Eigenvalues ---")
print(f"\n  Sum of eigenvalues = tr(S)")
print(f"    Σλᵢ = {sum(eigenvalues.real):.4f}")
print(f"    tr(S) = {trace_S:.4f}")
print(f"    Equal? {np.isclose(sum(eigenvalues.real), trace_S)}")

print(f"\n  Product of eigenvalues = det(S)")
print(f"    Πλᵢ = {np.prod(eigenvalues.real):.4f}")
print(f"    det(S) = {det_S:.4f}")
print(f"    Equal? {np.isclose(np.prod(eigenvalues.real), det_S)}")

# =============================================================================
# STEP 8: COVARIANCE MATRIX
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: COVARIANCE MATRIX")
print("=" * 70)

print("\n--- Definition ---")
print("\n  The covariance matrix captures relationships between variables:")
print("\n  Cov(X, Y) = E[(X - μx)(Y - μy)]")
print("\n  For a data matrix X (n samples × p features):")
print("  Σ = (1/(n-1)) × (X - X̄)ᵀ(X - X̄)")

# Calculate covariance matrix of the feature matrix
# Each column is a variable (grade band), each row is an observation (LO)
cov_matrix = np.cov(A, rowvar=False)

print(f"\n--- Covariance Matrix of Grade Band Features ---")
print(f"\n  Shape: {cov_matrix.shape}")
print(f"\n  Σ = ")
for i, row in enumerate(cov_matrix[:5, :5]):  # Show first 5x5
    print(f"    [{', '.join([f'{x:8.2f}' for x in row])}]")
if cov_matrix.shape[0] > 5:
    print("    ...")

print("\n--- Interpretation ---")
print("\n  Diagonal elements: Variance of each grade band")
for i, var in enumerate(np.diag(cov_matrix)[:5]):
    print(f"    Var(Grade {i+1}) = {var:.2f}")

print("\n  Off-diagonal elements: Covariance between grade bands")

# =============================================================================
# STEP 9: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Heatmap of feature matrix
ax1 = axes[0, 0]
im1 = ax1.imshow(A, cmap='YlOrRd', aspect='auto')
ax1.set_xlabel('Grade Band Index', fontsize=11)
ax1.set_ylabel('Learning Outcome', fontsize=11)
ax1.set_yticks(range(A.shape[0]))
ax1.set_yticklabels([f'LO{i+1}' for i in range(A.shape[0])])
ax1.set_xticks(range(A.shape[1]))
ax1.set_xticklabels([f'G{i+1}' for i in range(A.shape[1])], rotation=45)
ax1.set_title('Feature Matrix A\n(Word Counts per LO per Grade Band)', 
              fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Word Count')

# Add text annotations
for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        text = ax1.text(j, i, A[i, j], ha="center", va="center", 
                       color="white" if A[i, j] > A.max()/2 else "black", fontsize=8)

# Plot 2: Covariance matrix heatmap
ax2 = axes[0, 1]
im2 = ax2.imshow(cov_matrix, cmap='coolwarm', aspect='auto')
ax2.set_xlabel('Grade Band Index', fontsize=11)
ax2.set_ylabel('Grade Band Index', fontsize=11)
ax2.set_xticks(range(len(cov_matrix)))
ax2.set_xticklabels([f'G{i+1}' for i in range(len(cov_matrix))], rotation=45)
ax2.set_yticks(range(len(cov_matrix)))
ax2.set_yticklabels([f'G{i+1}' for i in range(len(cov_matrix))])
ax2.set_title('Covariance Matrix\nΣ = (Xᵀ X) / (n-1)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Covariance')

# Plot 3: Eigenvalue spectrum
ax3 = axes[1, 0]
sorted_eigenvalues = np.sort(eigenvalues.real)[::-1]
ax3.bar(range(len(sorted_eigenvalues)), sorted_eigenvalues, 
        color='steelblue', edgecolor='white', alpha=0.8)
ax3.set_xlabel('Eigenvalue Index', fontsize=11)
ax3.set_ylabel('Eigenvalue (λ)', fontsize=11)
ax3.set_title('Eigenvalue Spectrum of S = AAᵀ\n(Sorted in Descending Order)', 
              fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(sorted_eigenvalues)))
ax3.set_xticklabels([f'λ{i+1}' for i in range(len(sorted_eigenvalues))])
ax3.grid(True, alpha=0.3, axis='y')

# Add explained variance ratio
total = sum(sorted_eigenvalues)
cumulative = np.cumsum(sorted_eigenvalues) / total * 100
ax3_twin = ax3.twinx()
ax3_twin.plot(range(len(cumulative)), cumulative, 'ro-', linewidth=2, markersize=8)
ax3_twin.set_ylabel('Cumulative Variance (%)', color='red', fontsize=11)
ax3_twin.tick_params(axis='y', labelcolor='red')
ax3_twin.set_ylim(0, 105)

# Plot 4: Vector similarity matrix (cosine similarity)
ax4 = axes[1, 1]

# Calculate cosine similarity between all LO vectors
n_los = A.shape[0]
similarity_matrix = np.zeros((n_los, n_los))
for i in range(n_los):
    for j in range(n_los):
        norm_i = np.linalg.norm(A[i, :])
        norm_j = np.linalg.norm(A[j, :])
        if norm_i > 0 and norm_j > 0:
            similarity_matrix[i, j] = np.dot(A[i, :], A[j, :]) / (norm_i * norm_j)

im4 = ax4.imshow(similarity_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
ax4.set_xlabel('Learning Outcome', fontsize=11)
ax4.set_ylabel('Learning Outcome', fontsize=11)
ax4.set_xticks(range(n_los))
ax4.set_xticklabels([f'LO{i+1}' for i in range(n_los)])
ax4.set_yticks(range(n_los))
ax4.set_yticklabels([f'LO{i+1}' for i in range(n_los)])
ax4.set_title('Cosine Similarity Between LOs\ncos(θ) = (u·v)/(||u||×||v||)', 
              fontsize=12, fontweight='bold')
plt.colorbar(im4, ax=ax4, label='Cosine Similarity')

# Add text annotations
for i in range(n_los):
    for j in range(n_los):
        text = ax4.text(j, i, f'{similarity_matrix[i, j]:.2f}', 
                       ha="center", va="center", color="black", fontsize=9)

plt.tight_layout()
plt.savefig('../outputs/figures/linear_algebra_marking_matrix.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: linear_algebra_marking_matrix.png")

# =============================================================================
# STEP 10: APPLICATIONS IN DATA SCIENCE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: APPLICATIONS IN DATA SCIENCE")
print("=" * 70)

print("\n--- 1. Principal Component Analysis (PCA) ---")
print("\n  PCA uses eigendecomposition of the covariance matrix:")
print("  - Eigenvectors: Principal component directions")
print("  - Eigenvalues: Variance explained by each component")

explained_var_ratio = sorted_eigenvalues / sum(sorted_eigenvalues) * 100
print(f"\n  Variance explained by each component:")
for i, ratio in enumerate(explained_var_ratio):
    print(f"    PC{i+1}: {ratio:.2f}%")

print("\n--- 2. Linear Regression ---")
print("\n  Normal equations: β = (XᵀX)⁻¹Xᵀy")
print("  Uses matrix inverse to find optimal weights")

print("\n--- 3. Recommendation Systems ---")
print("\n  Matrix factorisation: R ≈ UV^T")
print("  Decomposes rating matrix into user and item factors")

print("\n--- 4. Neural Networks ---")
print("\n  Forward propagation: a = σ(Wx + b)")
print("  Weight matrices transform inputs through layers")

# =============================================================================
# STEP 11: SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 11: SUMMARY")
print("=" * 70)

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│                   LINEAR ALGEBRA SUMMARY                             │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│  Feature Matrix A:          {A.shape[0]} × {A.shape[1]}                            │")
print(f"│  Covariance Matrix Σ:       {cov_matrix.shape[0]} × {cov_matrix.shape[1]}                            │")
print(f"│  Gram Matrix AᵀA:           {ATA.shape[0]} × {ATA.shape[1]}                            │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│  Trace(AAᵀ):                {trace_S:>12.2f}                         │")
print(f"│  Determinant(AAᵀ):          {det_S:>12.2f}                         │")
print("├─────────────────────────────────────────────────────────────────────┤")
print(f"│  Dominant Eigenvalue:       {max(eigenvalues.real):>12.2f}                         │")
print(f"│  Variance by PC1:           {explained_var_ratio[0]:>11.2f}%                         │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\n--- Dataset Reference ---")
print("  Arden University (2024) COM7023 Mathematics for Data Science")
print("  Marking Matrix. Arden University.")

print("\n--- Further Reading ---")
print("  Strang, G. (2016) Introduction to Linear Algebra. 5th edn.")
print("  Wellesley-Cambridge Press.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF LINEAR ALGEBRA ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
