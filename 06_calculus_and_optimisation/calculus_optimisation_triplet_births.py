"""
================================================================================
CALCULUS AND OPTIMISATION - TRIPLET BIRTHS DATASET
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
This script applies calculus and optimisation concepts to the triplet births
dataset. We analyse rates of change in birth trends, perform numerical
integration for cumulative births, and use gradient descent for curve fitting.

MATHEMATICAL CONCEPTS:
- Numerical differentiation for trend analysis
- Integration for cumulative measures
- Cost functions for regression
- Gradient descent for parameter estimation
- Polynomial curve fitting optimisation

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
from scipy import integrate
from scipy.optimize import minimize, curve_fit

# =============================================================================
# CONFIGURATION
# =============================================================================

pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-whitegrid')
np.set_printoptions(precision=4, suppress=True)

# Define script directory for relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# HEADER INFORMATION
# =============================================================================

print("=" * 70)
print("CALCULUS AND OPTIMISATION")
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

print("\n--- Dataset Preview ---")
print(f"{'Year':>8}  {'Triplets':>12}")
print("-" * 25)
for i in range(min(10, n)):
    print(f"{int(years[i]):>8}  {triplets[i]:>12.0f}")
print("...")

# =============================================================================
# STEP 2: NUMERICAL DIFFERENTIATION - RATE OF CHANGE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: NUMERICAL DIFFERENTIATION - RATE OF CHANGE")
print("=" * 70)

print("\n--- Discrete Derivative Approximation ---")
print("\n  For discrete data points, we approximate the derivative using")
print("  finite differences:")
print("\n  Forward difference: f'(xᵢ) ≈ [f(xᵢ₊₁) - f(xᵢ)] / (xᵢ₊₁ - xᵢ)")
print("  Central difference: f'(xᵢ) ≈ [f(xᵢ₊₁) - f(xᵢ₋₁)] / (xᵢ₊₁ - xᵢ₋₁)")

# Calculate numerical derivatives
# Forward difference (for all but last point)
forward_diff = np.diff(triplets) / np.diff(years)

# Central difference (for interior points)
central_diff = np.zeros(n - 2)
for i in range(1, n - 1):
    central_diff[i - 1] = (triplets[i + 1] - triplets[i - 1]) / (years[i + 1] - years[i - 1])

print("\n--- Rate of Change (Forward Difference) ---")
print(f"\n  {'Year':>8}  {'Triplets':>12}  {'Rate of Change':>16}")
print("  " + "-" * 45)
for i in range(min(10, len(forward_diff))):
    print(f"  {int(years[i]):>8}  {triplets[i]:>12.0f}  {forward_diff[i]:>16.2f} per year")
print("  ...")

# Find years with maximum increase and decrease
max_increase_idx = np.argmax(forward_diff)
max_decrease_idx = np.argmin(forward_diff)

print(f"\n--- Key Findings ---")
print(f"\n  Maximum Increase:")
print(f"    Year: {int(years[max_increase_idx])} → {int(years[max_increase_idx + 1])}")
print(f"    Change: {triplets[max_increase_idx]:.0f} → {triplets[max_increase_idx + 1]:.0f}")
print(f"    Rate: +{forward_diff[max_increase_idx]:.2f} triplet deliveries per year")

print(f"\n  Maximum Decrease:")
print(f"    Year: {int(years[max_decrease_idx])} → {int(years[max_decrease_idx + 1])}")
print(f"    Change: {triplets[max_decrease_idx]:.0f} → {triplets[max_decrease_idx + 1]:.0f}")
print(f"    Rate: {forward_diff[max_decrease_idx]:.2f} triplet deliveries per year")

# =============================================================================
# STEP 3: SECOND DERIVATIVE - ACCELERATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: SECOND DERIVATIVE - ACCELERATION")
print("=" * 70)

print("\n--- Second Derivative Approximation ---")
print("\n  The second derivative measures how the rate of change is changing:")
print("  f''(x) ≈ [f'(xᵢ₊₁) - f'(xᵢ)] / h")
print("\n  Interpretation:")
print("    f''(x) > 0: Rate of change is increasing (accelerating)")
print("    f''(x) < 0: Rate of change is decreasing (decelerating)")

# Calculate second derivative
second_derivative = np.diff(forward_diff)

print("\n--- Acceleration of Triplet Births ---")
print(f"\n  {'Year':>8}  {'First Derivative':>18}  {'Second Derivative':>18}")
print("  " + "-" * 55)
for i in range(min(10, len(second_derivative))):
    print(f"  {int(years[i+1]):>8}  {forward_diff[i]:>18.2f}  {second_derivative[i]:>18.2f}")
print("  ...")

# Identify inflection points (where second derivative changes sign)
sign_changes = np.where(np.diff(np.sign(second_derivative)))[0]
print(f"\n--- Inflection Points (Change in Trend Direction) ---")
for idx in sign_changes[:5]:
    year_at_inflection = years[idx + 1]
    print(f"    Year {int(year_at_inflection)}: Trend changed from ",
          f"{'accelerating' if second_derivative[idx] > 0 else 'decelerating'} to ",
          f"{'decelerating' if second_derivative[idx] > 0 else 'accelerating'}")

# =============================================================================
# STEP 4: NUMERICAL INTEGRATION - CUMULATIVE BIRTHS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: NUMERICAL INTEGRATION - CUMULATIVE BIRTHS")
print("=" * 70)

print("\n--- Numerical Integration Methods ---")
print("\n  For discrete data, we use numerical integration methods:")
print("\n  Trapezoidal Rule:")
print("    ∫ f(x)dx ≈ (h/2) × [f(x₀) + 2f(x₁) + ... + 2f(xₙ₋₁) + f(xₙ)]")
print("\n  Simpson's Rule:")
print("    ∫ f(x)dx ≈ (h/3) × [f(x₀) + 4f(x₁) + 2f(x₂) + ... + f(xₙ)]")

# Trapezoidal integration
cumulative_trapezoidal = integrate.cumulative_trapezoid(triplets, years, initial=0)

# Total integral using different methods
total_trapezoidal = np.trapezoid(triplets, years)
total_simpson = integrate.simpson(triplets, x=years)

print("\n--- Cumulative Triplet Deliveries Over Time ---")
print(f"\n  {'Year':>8}  {'Annual':>12}  {'Cumulative':>15}")
print("  " + "-" * 45)
for i in range(0, n, 5):  # Every 5 years
    print(f"  {int(years[i]):>8}  {triplets[i]:>12.0f}  {cumulative_trapezoidal[i]:>15.0f}")

print(f"\n--- Total Integration Results ---")
print(f"\n  Period: {int(years[0])} - {int(years[-1])}")
print(f"  Trapezoidal Rule: {total_trapezoidal:,.0f} delivery-years")
print(f"  Simpson's Rule:   {total_simpson:,.0f} delivery-years")

print("\n  Note: These represent the 'area under the curve' of annual")
print("  triplet deliveries over time, a measure of cumulative impact.")

# Average annual births
average_annual = total_trapezoidal / (years[-1] - years[0])
print(f"\n  Average Annual Deliveries: {average_annual:.0f}")

# =============================================================================
# STEP 5: COST FUNCTION FOR REGRESSION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: COST FUNCTION FOR REGRESSION")
print("=" * 70)

print("\n--- Mean Squared Error (MSE) Cost Function ---")
print("\n  For linear regression y = mx + b, the cost function is:")
print("\n         1   n")
print("  J(m,b) = ─── Σ (yᵢ - (mxᵢ + b))²")
print("          2n i=1")
print("\n  Goal: Find m and b that minimise J(m,b)")

# Normalize years for numerical stability
years_normalized = years - years.mean()

def cost_function(params, x, y):
    """Calculate MSE cost function."""
    m, b = params
    y_pred = m * x + b
    return np.mean((y - y_pred) ** 2) / 2

def cost_gradient(params, x, y):
    """Calculate gradient of cost function."""
    m, b = params
    n = len(y)
    y_pred = m * x + b
    error = y_pred - y
    
    dm = np.mean(error * x)
    db = np.mean(error)
    
    return np.array([dm, db])

# Calculate initial cost
initial_params = [0, np.mean(triplets)]  # Start with horizontal line at mean
initial_cost = cost_function(initial_params, years_normalized, triplets)

print(f"\n  Initial Parameters:")
print(f"    m (slope) = {initial_params[0]}")
print(f"    b (intercept) = {initial_params[1]:.2f}")
print(f"    Initial Cost J(m,b) = {initial_cost:.2f}")

# =============================================================================
# STEP 6: GRADIENT DESCENT FOR LINEAR REGRESSION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: GRADIENT DESCENT FOR LINEAR REGRESSION")
print("=" * 70)

print("\n--- Gradient Descent Update Rules ---")
print("\n  m := m - α × ∂J/∂m")
print("  b := b - α × ∂J/∂b")
print("\n  where:")
print("    ∂J/∂m = (1/n) Σ (ŷᵢ - yᵢ) × xᵢ")
print("    ∂J/∂b = (1/n) Σ (ŷᵢ - yᵢ)")

def gradient_descent_linear(x, y, learning_rate, num_iterations):
    """
    Perform gradient descent for linear regression.
    
    Returns:
        history: List of (params, cost) tuples
    """
    m, b = 0.0, np.mean(y)
    history = [((m, b), cost_function([m, b], x, y))]
    
    for i in range(num_iterations):
        gradient = cost_gradient([m, b], x, y)
        m = m - learning_rate * gradient[0]
        b = b - learning_rate * gradient[1]
        cost = cost_function([m, b], x, y)
        history.append(((m, b), cost))
    
    return history

# Run gradient descent
learning_rate = 0.0001
num_iterations = 10000

history = gradient_descent_linear(years_normalized, triplets, 
                                   learning_rate, num_iterations)

print(f"\n  Gradient Descent Parameters:")
print(f"    Learning Rate (α): {learning_rate}")
print(f"    Iterations: {num_iterations}")

print(f"\n  Convergence History:")
print("  " + "-" * 55)
print(f"  {'Iteration':>10}  {'Slope (m)':>12}  {'Intercept (b)':>14}  {'Cost':>12}")
print("  " + "-" * 55)

iterations_to_show = [0, 1, 10, 100, 1000, 5000, 10000]
for i in iterations_to_show:
    if i < len(history):
        (m, b), cost = history[i]
        print(f"  {i:>10}  {m:>12.4f}  {b:>14.2f}  {cost:>12.2f}")

final_params, final_cost = history[-1]
print(f"\n  Final Result:")
print(f"    m* = {final_params[0]:.6f}")
print(f"    b* = {final_params[1]:.6f}")
print(f"    Final Cost = {final_cost:.4f}")

# Convert back to original scale
m_original = final_params[0]
b_original = final_params[1] - final_params[0] * (-years.mean())

print(f"\n  In Original Units:")
print(f"    ŷ = {m_original:.4f}x + {b_original:.4f}")
print(f"    (where x is the year)")

# =============================================================================
# STEP 7: ANALYTICAL SOLUTION COMPARISON
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: ANALYTICAL SOLUTION COMPARISON")
print("=" * 70)

print("\n--- Normal Equations Solution ---")
print("\n  The closed-form solution is:")
print("    m = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²")
print("    b = ȳ - m × x̄")

# Calculate analytical solution
x_mean = np.mean(years_normalized)
y_mean = np.mean(triplets)

m_analytical = np.sum((years_normalized - x_mean) * (triplets - y_mean)) / \
               np.sum((years_normalized - x_mean) ** 2)
b_analytical = y_mean - m_analytical * x_mean

print(f"\n  Analytical Solution:")
print(f"    m = {m_analytical:.6f}")
print(f"    b = {b_analytical:.6f}")

print(f"\n  Gradient Descent Solution:")
print(f"    m = {final_params[0]:.6f}")
print(f"    b = {final_params[1]:.6f}")

print(f"\n  Comparison:")
print(f"    Difference in m: {abs(m_analytical - final_params[0]):.6f}")
print(f"    Difference in b: {abs(b_analytical - final_params[1]):.6f}")

# =============================================================================
# STEP 8: POLYNOMIAL CURVE FITTING
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: POLYNOMIAL CURVE FITTING")
print("=" * 70)

print("\n--- Higher Degree Polynomials ---")
print("\n  Linear regression may not capture complex trends.")
print("  We can fit higher-degree polynomials:")
print("    y = a₀ + a₁x + a₂x² + a₃x³ + ...")

# Define polynomial models
def polynomial_model(x, *coeffs):
    """General polynomial model."""
    return sum(c * x**i for i, c in enumerate(coeffs))

# Fit polynomials of different degrees
degrees = [1, 2, 3, 4]
poly_results = {}

print("\n--- Polynomial Fit Comparison ---")
print("  " + "-" * 60)

for degree in degrees:
    # Fit polynomial
    coeffs = np.polyfit(years_normalized, triplets, degree)
    poly = np.poly1d(coeffs)
    y_pred = poly(years_normalized)
    
    # Calculate metrics
    ss_res = np.sum((triplets - y_pred) ** 2)
    ss_tot = np.sum((triplets - np.mean(triplets)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    mse = np.mean((triplets - y_pred) ** 2)
    
    poly_results[degree] = {
        'coeffs': coeffs,
        'poly': poly,
        'r_squared': r_squared,
        'mse': mse
    }
    
    print(f"\n  Degree {degree} Polynomial:")
    print(f"    R² = {r_squared:.6f}")
    print(f"    MSE = {mse:.2f}")
    print(f"    Coefficients: {coeffs}")

# Best model selection
best_degree = max(poly_results.keys(), key=lambda d: poly_results[d]['r_squared'])
print(f"\n  Best Fit (by R²): Degree {best_degree}")
print(f"    R² = {poly_results[best_degree]['r_squared']:.6f}")

# =============================================================================
# STEP 9: OPTIMISATION WITH SCIPY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: OPTIMISATION WITH SCIPY")
print("=" * 70)

print("\n--- Using scipy.optimize.minimize ---")
print("\n  SciPy provides advanced optimisation algorithms:")
print("    - BFGS (Broyden–Fletcher–Goldfarb–Shanno)")
print("    - L-BFGS-B (Limited-memory BFGS with bounds)")
print("    - Nelder-Mead (Simplex method)")

def total_cost(params, x, y):
    """Cost function for optimisation."""
    m, b = params
    y_pred = m * x + b
    return np.sum((y - y_pred) ** 2) / (2 * len(y))

# Run optimisation with different methods
methods = ['Nelder-Mead', 'BFGS', 'L-BFGS-B']
initial_guess = [0, np.mean(triplets)]

print("\n  Optimisation Results:")
print("  " + "-" * 65)

for method in methods:
    result = minimize(total_cost, initial_guess, args=(years_normalized, triplets),
                     method=method)
    print(f"\n  Method: {method}")
    print(f"    m = {result.x[0]:.6f}")
    print(f"    b = {result.x[1]:.6f}")
    print(f"    Final Cost = {result.fun:.6f}")
    print(f"    Iterations = {result.nit if hasattr(result, 'nit') else 'N/A'}")
    print(f"    Success = {result.success}")

# =============================================================================
# STEP 10: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Rate of change (derivative)
ax1 = axes[0, 0]
ax1.bar(years[:-1], forward_diff, width=0.8, alpha=0.7, 
        color=['green' if d >= 0 else 'red' for d in forward_diff],
        edgecolor='white')
ax1.axhline(y=0, color='black', linewidth=1, linestyle='-')
ax1.set_xlabel('Year', fontsize=11)
ax1.set_ylabel('Rate of Change (Δ Triplets/Year)', fontsize=11)
ax1.set_title('Numerical Derivative: Rate of Change\nf\'(x) ≈ [f(x+h) - f(x)] / h', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# Highlight max/min
ax1.scatter([years[max_increase_idx]], [forward_diff[max_increase_idx]], 
           color='darkgreen', s=100, zorder=5, marker='^',
           label=f'Max Increase: {forward_diff[max_increase_idx]:.0f}')
ax1.scatter([years[max_decrease_idx]], [forward_diff[max_decrease_idx]], 
           color='darkred', s=100, zorder=5, marker='v',
           label=f'Max Decrease: {forward_diff[max_decrease_idx]:.0f}')
ax1.legend()

# Plot 2: Cumulative integration
ax2 = axes[0, 1]
ax2.fill_between(years, 0, triplets, alpha=0.3, color='blue', 
                  label='Annual Deliveries')
ax2.plot(years, triplets, 'b-', linewidth=2)
ax2.plot(years, cumulative_trapezoidal / 100, 'r--', linewidth=2,
        label=f'Cumulative (÷100)\nTotal: {total_trapezoidal:,.0f}')
ax2.set_xlabel('Year', fontsize=11)
ax2.set_ylabel('Triplet Deliveries', fontsize=11)
ax2.set_title('Numerical Integration: Cumulative Sum\n∫f(x)dx ≈ Trapezoidal Rule', 
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Gradient descent convergence
ax3 = axes[1, 0]
costs = [h[1] for h in history]
ax3.plot(costs, 'b-', linewidth=1.5)
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('Cost J(m, b)', fontsize=11)
ax3.set_title('Gradient Descent Convergence\nMinimising MSE Cost Function', 
              fontsize=12, fontweight='bold')
ax3.set_yscale('log')
ax3.grid(True, alpha=0.3)

# Add text with final values
textstr = f'Final m = {final_params[0]:.4f}\nFinal b = {final_params[1]:.2f}\nFinal Cost = {final_cost:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax3.text(0.65, 0.95, textstr, transform=ax3.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

# Plot 4: Polynomial fits comparison
ax4 = axes[1, 1]
ax4.scatter(years, triplets, color='blue', s=30, alpha=0.7, 
            label='Actual Data', zorder=5)

# Plot polynomial fits
colors = ['orange', 'green', 'red', 'purple']
for degree, color in zip(degrees, colors):
    poly = poly_results[degree]['poly']
    y_pred = poly(years_normalized)
    r2 = poly_results[degree]['r_squared']
    ax4.plot(years, y_pred, color=color, linewidth=2, 
             label=f'Degree {degree} (R²={r2:.4f})')

ax4.set_xlabel('Year', fontsize=11)
ax4.set_ylabel('Triplet Deliveries', fontsize=11)
ax4.set_title('Polynomial Curve Fitting\nComparing Different Degrees', 
              fontsize=12, fontweight='bold')
ax4.legend(loc='upper right')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
# Ensure output directory exists
output_dir = os.path.join(script_dir, '../outputs/figures')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(script_dir, '../outputs/figures/calculus_optimisation_triplet_births.png'),
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: calculus_optimisation_triplet_births.png")

# =============================================================================
# STEP 11: SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 11: SUMMARY")
print("=" * 70)

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│              CALCULUS AND OPTIMISATION SUMMARY                       │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  DIFFERENTIATION                                                     │")
print(f"│    Maximum Rate Increase: {forward_diff[max_increase_idx]:>+8.2f} (Year {int(years[max_increase_idx])})      │")
print(f"│    Maximum Rate Decrease: {forward_diff[max_decrease_idx]:>+8.2f} (Year {int(years[max_decrease_idx])})      │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  INTEGRATION                                                         │")
print(f"│    Total (Trapezoidal): {total_trapezoidal:>12,.0f} delivery-years             │")
print(f"│    Average Annual:      {average_annual:>12,.0f} deliveries                   │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  GRADIENT DESCENT                                                    │")
print(f"│    Final Slope (m):     {final_params[0]:>12.6f}                              │")
print(f"│    Final Intercept (b): {final_params[1]:>12.2f}                              │")
print(f"│    Final Cost:          {final_cost:>12.4f}                              │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  POLYNOMIAL FITTING                                                  │")
print(f"│    Best Degree:         {best_degree:>12}                              │")
print(f"│    Best R²:             {poly_results[best_degree]['r_squared']:>12.6f}                              │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\n--- Dataset Reference ---")
print("  Human Multiple Births Database (2024) FRA_InputData_25.11.2024.xlsx.")
print("  Available at: https://www.twinbirths.org/en/data-metadata/")
print("  (Accessed: 25 November 2024).")

print("\n--- Further Reading ---")
print("  Stewart, J. (2015) Calculus: Early Transcendentals. 8th edn.")
print("  Cengage Learning.")

# =============================================================================
# END OF SCRIPT
# =============================================================================

print("\n" + "=" * 70)
print("END OF CALCULUS AND OPTIMISATION ANALYSIS")
print("Student: Farid Negahbnai (24154844)")
print("=" * 70)
