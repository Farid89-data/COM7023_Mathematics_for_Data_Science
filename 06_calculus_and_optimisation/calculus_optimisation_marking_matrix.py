"""
================================================================================
CALCULUS AND OPTIMISATION - MARKING MATRIX DATASET
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
This script demonstrates calculus concepts using the marking matrix dataset.
We explore derivatives for rate of change analysis, integrals for cumulative
measures, and optimisation for finding optimal grade distributions.

MATHEMATICAL CONCEPTS:
- Limits and continuity
- Derivatives and differentiation rules
- Partial derivatives
- Integrals and the Fundamental Theorem
- Optimisation and critical points
- Gradient descent algorithm

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
from scipy import integrate
from scipy.optimize import minimize

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
print("CALCULUS AND OPTIMISATION")
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

grade_midpoints = np.array([95, 85, 74.5, 64.5, 54.5, 44.5, 34.5, 24.5, 9.5])

# Create complexity scores based on word counts
complexity_scores = []
for idx, row in df.iterrows():
    total_words = 0
    for grade_col in grade_columns:
        if grade_col in df.columns:
            text = str(row[grade_col])
            total_words += len(text.split())
    complexity_scores.append(total_words)

complexity_scores = np.array(complexity_scores)

print(f"\nNumber of Learning Outcomes: {len(df)}")
print(f"Number of Grade Bands: {len(grade_columns)}")
print(f"Complexity Scores: {complexity_scores}")

# =============================================================================
# STEP 2: LIMITS - THE FOUNDATION OF CALCULUS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 2: LIMITS - THE FOUNDATION OF CALCULUS")
print("=" * 70)

print("\n--- What is a Limit? ---")
print("\n  A limit describes the value a function approaches as the input")
print("  approaches a particular value:")
print("\n                    lim f(x) = L")
print("                   x → a")
print("\n  This is read as: 'the limit of f(x) as x approaches a equals L'")

print("\n--- Numerical Demonstration ---")
print("\n  Consider the function f(x) = (x² - 1) / (x - 1)")
print("  This is undefined at x = 1, but we can find the limit:")

def f_limit_demo(x):
    """Function to demonstrate limit calculation."""
    if x == 1:
        return np.nan
    return (x**2 - 1) / (x - 1)

print("\n  Approaching x = 1 from the left:")
print("  " + "-" * 40)
left_values = [0.9, 0.99, 0.999, 0.9999, 0.99999]
for x in left_values:
    print(f"    f({x}) = {f_limit_demo(x):.6f}")

print("\n  Approaching x = 1 from the right:")
print("  " + "-" * 40)
right_values = [1.1, 1.01, 1.001, 1.0001, 1.00001]
for x in right_values:
    print(f"    f({x}) = {f_limit_demo(x):.6f}")

print("\n  Conclusion: lim(x→1) (x² - 1)/(x - 1) = 2")
print("\n  Algebraically: (x² - 1)/(x - 1) = (x+1)(x-1)/(x-1) = x + 1")
print("                 At x = 1: x + 1 = 2 ✓")

# =============================================================================
# STEP 3: DERIVATIVES - RATE OF CHANGE
# =============================================================================

print("\n" + "=" * 70)
print("STEP 3: DERIVATIVES - RATE OF CHANGE")
print("=" * 70)

print("\n--- Definition of the Derivative ---")
print("\n  The derivative is defined as the limit of the difference quotient:")
print("\n            f(x + h) - f(x)")
print("  f'(x) = lim ─────────────────")
print("          h→0        h")

print("\n--- Numerical Differentiation ---")
print("\n  We can approximate derivatives numerically using finite differences:")
print("\n  Forward difference:  f'(x) ≈ [f(x+h) - f(x)] / h")
print("  Central difference:  f'(x) ≈ [f(x+h) - f(x-h)] / (2h)")

# Create a grade distribution function (modeled as quadratic)
def grade_distribution(x, a=0.001, b=-0.1, c=10):
    """Quadratic model for grade distribution density."""
    return a * x**2 + b * x + c

def grade_distribution_derivative(x, a=0.001, b=-0.1):
    """Analytical derivative of grade distribution."""
    return 2 * a * x + b

def numerical_derivative(f, x, h=0.0001):
    """Calculate numerical derivative using central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)

print("\n--- Example: Grade Distribution Function ---")
print("\n  f(x) = 0.001x² - 0.1x + 10  (where x is grade percentage)")
print("  f'(x) = 0.002x - 0.1")

print("\n  Comparing Analytical and Numerical Derivatives:")
print("  " + "-" * 55)
print(f"  {'Grade (x)':>12}  {'f(x)':>12}  {'f\'(x) Analytical':>18}  {'f\'(x) Numerical':>18}")
print("  " + "-" * 55)

test_grades = [20, 40, 50, 60, 80, 100]
for x in test_grades:
    fx = grade_distribution(x)
    analytical = grade_distribution_derivative(x)
    numerical = numerical_derivative(grade_distribution, x)
    print(f"  {x:>12}  {fx:>12.4f}  {analytical:>18.6f}  {numerical:>18.6f}")

# =============================================================================
# STEP 4: DIFFERENTIATION RULES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 4: DIFFERENTIATION RULES")
print("=" * 70)

print("\n" + "-" * 50)
print("4.1 POWER RULE")
print("-" * 50)

print("\n  Rule: d/dx[xⁿ] = n·xⁿ⁻¹")
print("\n  Examples:")
print("    d/dx[x²] = 2x")
print("    d/dx[x³] = 3x²")
print("    d/dx[x⁻¹] = -x⁻²")
print("    d/dx[√x] = d/dx[x^(1/2)] = (1/2)x^(-1/2)")

# Demonstrate with grade midpoints
print("\n  Application to grade midpoints (as power function):")
print(f"  If f(x) = x², then f'(x) = 2x")
for i, midpoint in enumerate(grade_midpoints[:4]):
    print(f"    At x = {midpoint}: f'({midpoint}) = 2 × {midpoint} = {2*midpoint}")

print("\n" + "-" * 50)
print("4.2 SUM AND DIFFERENCE RULE")
print("-" * 50)

print("\n  Rule: d/dx[f(x) ± g(x)] = f'(x) ± g'(x)")
print("\n  Example: d/dx[x² + 3x - 5] = 2x + 3")

print("\n" + "-" * 50)
print("4.3 PRODUCT RULE")
print("-" * 50)

print("\n  Rule: d/dx[f(x)·g(x)] = f'(x)·g(x) + f(x)·g'(x)")
print("\n  Example: d/dx[x²·eˣ] = 2x·eˣ + x²·eˣ = eˣ(x² + 2x)")

def product_demo(x):
    """f(x) = x² · e^x"""
    return x**2 * np.exp(x)

def product_demo_derivative(x):
    """Analytical derivative using product rule."""
    return np.exp(x) * (x**2 + 2*x)

print("\n  Verification:")
x_test = 2
analytical = product_demo_derivative(x_test)
numerical = numerical_derivative(product_demo, x_test)
print(f"    At x = {x_test}:")
print(f"    Analytical: {analytical:.6f}")
print(f"    Numerical:  {numerical:.6f}")

print("\n" + "-" * 50)
print("4.4 CHAIN RULE")
print("-" * 50)

print("\n  Rule: d/dx[f(g(x))] = f'(g(x)) · g'(x)")
print("\n  Example: d/dx[sin(x²)] = cos(x²) · 2x")

print("\n  This is crucial for neural network backpropagation!")

def chain_demo(x):
    """f(x) = sin(x²)"""
    return np.sin(x**2)

def chain_demo_derivative(x):
    """Analytical derivative using chain rule."""
    return np.cos(x**2) * 2 * x

print("\n  Verification:")
x_test = 1.5
analytical = chain_demo_derivative(x_test)
numerical = numerical_derivative(chain_demo, x_test)
print(f"    At x = {x_test}:")
print(f"    d/dx[sin(x²)] = cos(x²) · 2x")
print(f"    Analytical: {analytical:.6f}")
print(f"    Numerical:  {numerical:.6f}")

# =============================================================================
# STEP 5: PARTIAL DERIVATIVES
# =============================================================================

print("\n" + "=" * 70)
print("STEP 5: PARTIAL DERIVATIVES")
print("=" * 70)

print("\n--- Multivariate Functions ---")
print("\n  For functions of multiple variables, partial derivatives measure")
print("  the rate of change with respect to one variable while holding")
print("  others constant.")
print("\n  Notation: ∂f/∂x (partial derivative of f with respect to x)")

print("\n--- Example: Performance Function ---")
print("\n  Let P(s, e) = 2s² + 3e² - se + 10")
print("  where s = study hours, e = experience level")
print("\n  Partial derivatives:")
print("    ∂P/∂s = 4s - e    (rate of change with respect to study hours)")
print("    ∂P/∂e = 6e - s    (rate of change with respect to experience)")

def performance(s, e):
    """Performance function P(s, e)."""
    return 2*s**2 + 3*e**2 - s*e + 10

def partial_s(s, e):
    """Partial derivative with respect to s."""
    return 4*s - e

def partial_e(s, e):
    """Partial derivative with respect to e."""
    return 6*e - s

print("\n  Evaluation at (s=3, e=2):")
s, e = 3, 2
print(f"    P(3, 2) = {performance(s, e)}")
print(f"    ∂P/∂s at (3, 2) = 4(3) - 2 = {partial_s(s, e)}")
print(f"    ∂P/∂e at (3, 2) = 6(2) - 3 = {partial_e(s, e)}")

print("\n--- The Gradient ---")
print("\n  The gradient is a vector of all partial derivatives:")
print("    ∇P = [∂P/∂s, ∂P/∂e]ᵀ")
print(f"\n    ∇P at (3, 2) = [{partial_s(s, e)}, {partial_e(s, e)}]ᵀ")
print("\n  The gradient points in the direction of steepest increase.")

# =============================================================================
# STEP 6: INTEGRALS - ACCUMULATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 6: INTEGRALS - ACCUMULATION")
print("=" * 70)

print("\n--- Definite Integral ---")
print("\n  The definite integral represents the signed area under a curve:")
print("\n         b")
print("        ∫ f(x) dx = F(b) - F(a)")
print("         a")
print("\n  where F'(x) = f(x) (F is the antiderivative of f)")

print("\n--- Fundamental Theorem of Calculus ---")
print("\n  Part 1: If F(x) = ∫ₐˣ f(t) dt, then F'(x) = f(x)")
print("  Part 2: ∫ₐᵇ f(x) dx = F(b) - F(a)")

print("\n--- Example: Integrating the Grade Distribution ---")
print("\n  f(x) = 0.001x² - 0.1x + 10")
print("  F(x) = ∫ f(x) dx = (0.001/3)x³ - (0.1/2)x² + 10x + C")
print("       = 0.000333x³ - 0.05x² + 10x + C")

def antiderivative(x):
    """Antiderivative of the grade distribution function."""
    return (0.001/3) * x**3 - (0.1/2) * x**2 + 10 * x

# Calculate integral from 40 to 70 (passing grade range)
a, b = 40, 70
analytical_integral = antiderivative(b) - antiderivative(a)

# Numerical integration using scipy
numerical_integral, error = integrate.quad(grade_distribution, a, b)

print(f"\n  Calculate ∫₄₀⁷⁰ f(x) dx:")
print(f"\n    Analytical: F(70) - F(40)")
print(f"              = {antiderivative(b):.4f} - {antiderivative(a):.4f}")
print(f"              = {analytical_integral:.4f}")
print(f"\n    Numerical (scipy.integrate.quad): {numerical_integral:.4f}")
print(f"    Numerical error estimate: {error:.2e}")

print("\n--- Application: Expected Value ---")
print("\n  For a probability density function (PDF), the expected value is:")
print("    E[X] = ∫ x · f(x) dx")

# =============================================================================
# STEP 7: OPTIMISATION - FINDING EXTREMA
# =============================================================================

print("\n" + "=" * 70)
print("STEP 7: OPTIMISATION - FINDING EXTREMA")
print("=" * 70)

print("\n--- Critical Points ---")
print("\n  A critical point occurs where f'(x) = 0 or f'(x) is undefined.")
print("\n  To classify critical points:")
print("    - f''(x) > 0: Local minimum (concave up)")
print("    - f''(x) < 0: Local maximum (concave down)")
print("    - f''(x) = 0: Inconclusive (use higher-order tests)")

print("\n--- Example: Finding Minimum of Cost Function ---")
print("\n  Let C(x) = x² - 6x + 15  (cost as function of some parameter)")
print("  C'(x) = 2x - 6")
print("  C''(x) = 2")
print("\n  Setting C'(x) = 0:")
print("    2x - 6 = 0")
print("    x = 3")
print("\n  Since C''(3) = 2 > 0, x = 3 is a local minimum.")

def cost_function(x):
    return x**2 - 6*x + 15

def cost_derivative(x):
    return 2*x - 6

def cost_second_derivative(x):
    return 2

# Verify numerically
critical_point = 3
print(f"\n  Verification:")
print(f"    C(3) = {cost_function(critical_point)}")
print(f"    C'(3) = {cost_derivative(critical_point)} (should be 0)")
print(f"    C''(3) = {cost_second_derivative(critical_point)} > 0 (minimum)")

# =============================================================================
# STEP 8: GRADIENT DESCENT ALGORITHM
# =============================================================================

print("\n" + "=" * 70)
print("STEP 8: GRADIENT DESCENT ALGORITHM")
print("=" * 70)

print("\n--- Algorithm Description ---")
print("\n  Gradient descent is an iterative optimisation algorithm:")
print("\n    θₜ₊₁ = θₜ - α · ∇J(θₜ)")
print("\n  where:")
print("    θ = parameters to optimise")
print("    α = learning rate (step size)")
print("    ∇J = gradient of the cost function")

print("\n--- Implementing Gradient Descent ---")

def objective_function(x):
    """Function to minimise: f(x) = (x - 3)² + 2"""
    return (x - 3)**2 + 2

def objective_gradient(x):
    """Gradient of objective function: f'(x) = 2(x - 3)"""
    return 2 * (x - 3)

def gradient_descent(gradient_func, initial_x, learning_rate, num_iterations):
    """
    Perform gradient descent optimisation.
    
    Parameters:
        gradient_func: Function to compute gradient
        initial_x: Starting point
        learning_rate: Step size (alpha)
        num_iterations: Number of iterations
    
    Returns:
        history: List of (x, f(x)) tuples
    """
    x = initial_x
    history = [(x, objective_function(x))]
    
    for i in range(num_iterations):
        gradient = gradient_func(x)
        x = x - learning_rate * gradient
        history.append((x, objective_function(x)))
    
    return history

# Run gradient descent
initial_x = 10
learning_rate = 0.1
num_iterations = 50

history = gradient_descent(objective_gradient, initial_x, learning_rate, num_iterations)

print(f"\n  Objective Function: f(x) = (x - 3)² + 2")
print(f"  True Minimum: x = 3, f(3) = 2")
print(f"\n  Gradient Descent Parameters:")
print(f"    Initial x: {initial_x}")
print(f"    Learning Rate (α): {learning_rate}")
print(f"    Iterations: {num_iterations}")

print(f"\n  Iteration History:")
print("  " + "-" * 50)
print(f"  {'Iteration':>10}  {'x':>12}  {'f(x)':>12}  {'Gradient':>12}")
print("  " + "-" * 50)

iterations_to_show = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
for i in iterations_to_show:
    if i < len(history):
        x, fx = history[i]
        grad = objective_gradient(x)
        print(f"  {i:>10}  {x:>12.6f}  {fx:>12.6f}  {grad:>12.6f}")

final_x, final_fx = history[-1]
print(f"\n  Final Result:")
print(f"    x* = {final_x:.6f} (true minimum: 3)")
print(f"    f(x*) = {final_fx:.6f} (true minimum: 2)")
print(f"    Error: |x* - 3| = {abs(final_x - 3):.6f}")

# =============================================================================
# STEP 9: LEARNING RATE ANALYSIS
# =============================================================================

print("\n" + "=" * 70)
print("STEP 9: LEARNING RATE ANALYSIS")
print("=" * 70)

print("\n--- Effect of Learning Rate ---")
print("\n  The learning rate α significantly affects convergence:")
print("    - Too small: Slow convergence")
print("    - Too large: May overshoot or diverge")
print("    - Just right: Fast, stable convergence")

learning_rates = [0.01, 0.1, 0.5, 0.9, 1.0]
results = {}

print("\n  Comparing Different Learning Rates:")
print("  " + "-" * 60)

for lr in learning_rates:
    history = gradient_descent(objective_gradient, initial_x, lr, 50)
    final_x, final_fx = history[-1]
    
    # Check for divergence
    if abs(final_x) > 1e6:
        status = "DIVERGED"
    elif abs(final_x - 3) < 0.001:
        status = "Converged"
    else:
        status = "Slow convergence"
    
    results[lr] = history
    print(f"    α = {lr}: x* = {final_x:>12.6f}, f(x*) = {final_fx:>12.6f} [{status}]")

# =============================================================================
# STEP 10: MULTIVARIATE GRADIENT DESCENT
# =============================================================================

print("\n" + "=" * 70)
print("STEP 10: MULTIVARIATE GRADIENT DESCENT")
print("=" * 70)

print("\n--- Extending to Multiple Variables ---")
print("\n  For f(x, y), the gradient is:")
print("    ∇f = [∂f/∂x, ∂f/∂y]ᵀ")
print("\n  Update rule:")
print("    [x, y]ₜ₊₁ = [x, y]ₜ - α · ∇f([x, y]ₜ)")

def rosenbrock(params):
    """Rosenbrock function (common optimisation test function)."""
    x, y = params
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(params):
    """Gradient of Rosenbrock function."""
    x, y = params
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def multivariate_gradient_descent(grad_func, obj_func, initial_params, 
                                   learning_rate, num_iterations):
    """Gradient descent for multivariate functions."""
    params = np.array(initial_params, dtype=float)
    history = [(params.copy(), obj_func(params))]
    
    for i in range(num_iterations):
        gradient = grad_func(params)
        params = params - learning_rate * gradient
        history.append((params.copy(), obj_func(params)))
    
    return history

# Run multivariate gradient descent on Rosenbrock function
initial_params = [-1, 1]
lr_multi = 0.001
iterations_multi = 10000

history_multi = multivariate_gradient_descent(
    rosenbrock_gradient, rosenbrock, initial_params, lr_multi, iterations_multi
)

print(f"\n  Rosenbrock Function: f(x,y) = (1-x)² + 100(y-x²)²")
print(f"  Global Minimum: (x, y) = (1, 1), f(1,1) = 0")
print(f"\n  Initial Point: {initial_params}")
print(f"  Learning Rate: {lr_multi}")
print(f"  Iterations: {iterations_multi}")

final_params, final_value = history_multi[-1]
print(f"\n  Final Result:")
print(f"    (x*, y*) = ({final_params[0]:.6f}, {final_params[1]:.6f})")
print(f"    f(x*, y*) = {final_value:.6f}")

# =============================================================================
# STEP 11: VISUALISATION
# =============================================================================

print("\n" + "=" * 70)
print("STEP 11: VISUALISATION")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Plot 1: Derivative visualisation
ax1 = axes[0, 0]
x = np.linspace(0, 100, 500)
y = grade_distribution(x)
y_prime = grade_distribution_derivative(x)

ax1.plot(x, y, 'b-', linewidth=2, label='f(x) = 0.001x² - 0.1x + 10')
ax1.plot(x, y_prime * 10 + 5, 'r--', linewidth=2, 
         label="f'(x) = 0.002x - 0.1 (scaled)")

# Add tangent line at x=50
x0 = 50
y0 = grade_distribution(x0)
slope = grade_distribution_derivative(x0)
tangent_x = np.linspace(30, 70, 100)
tangent_y = slope * (tangent_x - x0) + y0
ax1.plot(tangent_x, tangent_y, 'g-', linewidth=2, alpha=0.7, 
         label=f'Tangent at x=50 (slope={slope:.3f})')
ax1.scatter([x0], [y0], color='green', s=100, zorder=5)

ax1.set_xlabel('Grade (%)', fontsize=11)
ax1.set_ylabel('f(x)', fontsize=11)
ax1.set_title('Derivatives: Rate of Change\nTangent Line Shows Instantaneous Rate', 
              fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Plot 2: Integration visualisation
ax2 = axes[0, 1]
x = np.linspace(0, 100, 500)
y = grade_distribution(x)

ax2.plot(x, y, 'b-', linewidth=2, label='f(x)')
ax2.fill_between(x[(x >= 40) & (x <= 70)], 0, 
                  y[(x >= 40) & (x <= 70)], alpha=0.3, color='blue',
                  label=f'∫₄₀⁷⁰ f(x)dx = {analytical_integral:.2f}')
ax2.axvline(x=40, color='green', linestyle='--', alpha=0.7)
ax2.axvline(x=70, color='green', linestyle='--', alpha=0.7)

ax2.set_xlabel('Grade (%)', fontsize=11)
ax2.set_ylabel('f(x)', fontsize=11)
ax2.set_title('Integration: Area Under the Curve\n(Passing Grade Range: 40-70%)', 
              fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Gradient descent convergence
ax3 = axes[1, 0]
colors = plt.cm.viridis(np.linspace(0, 1, len(learning_rates)))

for lr, color in zip(learning_rates, colors):
    history = results[lr]
    x_vals = [h[0] for h in history[:30]]  # First 30 iterations
    ax3.plot(range(len(x_vals)), x_vals, '-o', markersize=3, 
             color=color, label=f'α = {lr}')

ax3.axhline(y=3, color='red', linestyle='--', linewidth=2, 
            label='Optimal x* = 3')
ax3.set_xlabel('Iteration', fontsize=11)
ax3.set_ylabel('x value', fontsize=11)
ax3.set_title('Gradient Descent: Learning Rate Comparison\nθₜ₊₁ = θₜ - α·∇J(θₜ)', 
              fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-5, 15)

# Plot 4: Cost function landscape with gradient descent path
ax4 = axes[1, 1]
x = np.linspace(-2, 10, 500)
y = objective_function(x)

ax4.plot(x, y, 'b-', linewidth=2, label='f(x) = (x-3)² + 2')

# Plot gradient descent path with α = 0.1
history = results[0.1]
x_path = [h[0] for h in history]
y_path = [h[1] for h in history]

ax4.scatter(x_path, y_path, c=range(len(x_path)), cmap='Reds', 
            s=50, zorder=5, label='Gradient Descent Path')
ax4.scatter([3], [2], color='green', s=200, marker='*', 
            zorder=6, label='Minimum (3, 2)')

# Add arrows showing direction
for i in range(0, min(5, len(x_path)-1)):
    ax4.annotate('', xy=(x_path[i+1], y_path[i+1]), 
                xytext=(x_path[i], y_path[i]),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

ax4.set_xlabel('x', fontsize=11)
ax4.set_ylabel('f(x)', fontsize=11)
ax4.set_title('Gradient Descent on Cost Function\nFollowing Negative Gradient Direction', 
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../outputs/figures/calculus_optimisation_marking_matrix.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print("\nVisualisation saved: calculus_optimisation_marking_matrix.png")

# =============================================================================
# STEP 12: SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("STEP 12: SUMMARY")
print("=" * 70)

print("\n┌─────────────────────────────────────────────────────────────────────┐")
print("│              CALCULUS AND OPTIMISATION SUMMARY                       │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  DERIVATIVES                                                         │")
print("│    Power Rule: d/dx[xⁿ] = n·xⁿ⁻¹                                    │")
print("│    Chain Rule: d/dx[f(g(x))] = f'(g(x))·g'(x)                       │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  INTEGRALS                                                           │")
print(f"│    ∫₄₀⁷⁰ f(x)dx = {analytical_integral:>10.4f}                                       │")
print("├─────────────────────────────────────────────────────────────────────┤")
print("│  GRADIENT DESCENT                                                    │")
print("│    Update: θₜ₊₁ = θₜ - α·∇J(θₜ)                                     │")
print(f"│    Best Learning Rate: α = 0.1                                      │")
print(f"│    Final x* = {results[0.1][-1][0]:>10.6f} (target: 3.0)                         │")
print("└─────────────────────────────────────────────────────────────────────┘")

print("\n--- Dataset Reference ---")
print("  Arden University (2024) COM7023 Mathematics for Data Science")
print("  Marking Matrix. Arden University.")

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
