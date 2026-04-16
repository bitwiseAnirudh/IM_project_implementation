# INDUSTRIAL MANAGEMENT
# PROJECT
# IMPLEMENTATION OF RESEARCH PAPER

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# First, we shall obtain the weights of the individual objective functions using AHP
# AHP (ANALYTIC HIERARCHY PROCESS)

# Step1: Define the matrix B
# Entries of B perform pairwise conparison for the relative importance of objective functions

# The rows and columns represent: [Sales, Costs, Productivity]
# According to the paper, only allowed values are 1,2,...9 and their reciprocals

B = np.array([
    [1.0, 1/5, 1/3],  # Sales comparisons
    [5.0, 1.0, 5.0],  # Costs comparisons
    [3.0, 1/5, 1.0]   # Productivity comparisons
])

# Step2: Finding eigenvalues and eigenvectors of B matrix
# np.linalg.eig calculates the roots of the characteristic equation
# It returns array of eigenvalues and matrix of corresponding eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(B)

# Step3: Find the max eigenvalue and corresponding eigenvector
# AHP theory states the true weights belong to the absolute maximum eigenvalue

max_idx = np.argmax(np.real(eigenvalues))
lambda_max = np.real(eigenvalues[max_idx])
raw_eigvec = np.real(eigenvectors[:, max_idx])

# we are using the entries as weights so we want it to sum up to 1
ahp_weights = raw_eigvec / np.sum(raw_eigvec)

print("AHP result")
print(f"Maximum Eigenvalue (Lambda Max): {lambda_max:.3f}")
print(f"Weight for sales:     {ahp_weights[0]:.3f}")
print(f"Weight for costs:  {ahp_weights[1]:.3f}")
print(f"Weight for productivity:    {ahp_weights[2]:.3f}")

# these values for eigenvector do not match with the paper.
# for this matrrix, the paper claims (0.132, 0.612, 0.256) but we obtained (0.097, 0.701, 0.202).

# INDIVIDUAL OPTIMIZAITON
# 1. MAXIMIZING TURNOVER

# constraints
# Sales volume achieved in thousand-RON for 12 months/ coefficients of objective function
C = np.array([1438, 1309, 1157, 1811, 1089, 1078, 
              1088, 1025, 1003, 1200, 1117,  927])

# Actual demand/ customer orders for 12 months
D = np.array([2548556, 2550855, 2735389, 2503787, 2750643, 2624632, 
              2682187, 2563748, 2451276, 2992994, 2761731, 2258991])

# Max production capacity for 12 months
Cap = np.array([2700000, 2500000, 2700000, 2800000, 2800000, 2700000, 
                2800000, 3000000, 2800000, 3000000, 2800000, 2100000])

# SciPy minimizes by default, so we mathematically invert the goal: Minimize (-C * X)
objective_function = -C

# the factory must produce at least the minimum customer orders each month
# we set this as a strict boundary for each variable X
X_bounds = [(D[j], None) for j in range(12)]

# cumulative capacity constraint
Sigma_matrix = np.tril(np.ones((12, 12)))
Sigma_capacity = np.cumsum(Cap)

# solve
res = linprog(c=objective_function, 
              A_ub=Sigma_matrix, 
              b_ub=Sigma_capacity, 
              bounds=X_bounds, 
              method='highs')

# display results
print("1) INDIVIDUAL OPTIMIZATION: MAXIMIZE TURNOVER")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

if res.success:
    X_sales = res.x
    for month, production_amount in zip(months, res.x):
        print(f"{month}: {round(production_amount)} pieces")

else:
    print("Optimization failed:", res.message)
print("\n")


# 2. MINIMIZING COST

# constraints
# cost objective function coefficients for 12 months
C_cost = np.array([80, 75, 70, 65, 60, 55, 
                   50, 45, 40, 35, 30, 25])

D = np.array([2548556, 2550855, 2735389, 2503787, 2750643, 2624632, 
              2682187, 2563748, 2451276, 2992994, 2761731, 2258991])

Cap = np.array([2700000, 2500000, 2700000, 2800000, 2800000, 2700000, 
                2800000, 3000000, 2800000, 3000000, 2800000, 2100000])


objective_function = C_cost

# cumulative capacity upper bound
Sigma_matrix = np.tril(np.ones((12, 12)))
Sigma_capacity = np.cumsum(Cap)
Negative_Sigma_matrix = -Sigma_matrix
Negative_Sigma_demand = -np.cumsum(D)

A_ub = np.vstack((Sigma_matrix, Negative_Sigma_matrix))
b_ub = np.concatenate((Sigma_capacity, Negative_Sigma_demand))

X_bounds = [(0, Cap[j]) for j in range(12)]

# solve
res = linprog(c=objective_function, 
              A_ub=A_ub, 
              b_ub=b_ub, 
              bounds=X_bounds, 
              method='highs')

# display results
print("INDIVIDUAL OPTIMIZATION: MINIMIZE COSTS")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

if res.success:
    X_cost = res.x
    for month, production_amount in zip(months, res.x):
        print(f"{month}: {round(production_amount):10,d} pieces")

else:
    print("Optimization failed:", res.message)
print("\n")


# 3. MAXIMIZING PRODUCTIVITY

# constraints
# productivity: pieces/worker
C_prod = np.array([9664, 10913, 12121, 11072, 11617, 11492, 
                   11706, 11122, 10939, 13438, 12178, 10341])

D = np.array([2548556, 2550855, 2735389, 2503787, 2750643, 2624632, 
              2682187, 2563748, 2451276, 2992994, 2761731, 2258991])

Cap = np.array([2700000, 2500000, 2700000, 2800000, 2800000, 2700000, 
                2800000, 3000000, 2800000, 3000000, 2800000, 2100000])


# we are maximizing so we mathematically invert the objective for SciPy
objective_function = -C_prod

# again, cumulative capacity
Sigma_matrix = np.tril(np.ones((12, 12)))
Sigma_capacity = np.cumsum(Cap)

X_bounds = [(D[j], None) for j in range(12)]

# solve
res = linprog(c=objective_function, 
              A_ub=Sigma_matrix, 
              b_ub=Sigma_capacity, 
              bounds=X_bounds, 
              method='highs')


print("INDIVIDUAL OPTIMIZATION: MAXIMIZE PRODUCTIVITY")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

if res.success:
    X_productivity = res.x
    for month, production_amount in zip(months, res.x):
        print(f"{month}: {round(production_amount):10,d}")
else:
    print("Optimization failed:", res.message)
print("\n")


# COmbined optimization using AHP obtained weights
w_sales = 0.132
w_costs = 0.612
w_prod  = 0.256

# The Raw Objective Coefficients
C_sales = np.array([1438, 1309, 1157, 1811, 1089, 1078, 1088, 1025, 1003, 1200, 1117,  927])
C_cost  = np.array([80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25])
C_prod  = np.array([9664, 10913, 12121, 11072, 11617, 11492, 11706, 11122, 10939, 13438, 12178, 10341])

D = np.array([2548556, 2550855, 2735389, 2503787, 2750643, 2624632, 
              2682187, 2563748, 2451276, 2992994, 2761731, 2258991])

Cap = np.array([2700000, 2500000, 2700000, 2800000, 2800000, 2700000, 
                2800000, 3000000, 2800000, 3000000, 2800000, 2100000])

# making the multi objective function using the weights
C_master = (w_sales * C_sales*1000) - (w_costs * C_cost) + (w_prod * C_prod)
# the following values are the paper's coefficients
C_master_flawed = np.array([
    193828.1,   # X1 (Jan)
    176990.6,   # X2 (Feb)
    157106.7,   # X3 (Mar)
    243037.9,   # X4 (Apr)
    147743.3,   # X5 (May)
    146130.1,   # X6 (Jun) --- Note: June lower than May
    147214.4,   # X7 (Jul)
    138780.9,   # X8 (Aug)
    135700.9,   # X9 (Sep)
    138455.58,  # X10 (Oct) --- The error: Matheatically this should be ~162,215
    150807.7,   # X11 (Nov)
    125128.3    # X12 (Dec)
])
objective_function = -C_master_flawed



# combined constraints
Sigma_matrix = np.tril(np.ones((12, 12)))
A_ub = np.vstack((Sigma_matrix, -Sigma_matrix))
b_ub = np.concatenate((np.cumsum(Cap), -np.cumsum(D)))
X_bounds = [(0, None) for j in range(12)]
X_bounds[9] = (0, 3000000)

# solve
res = linprog(c=objective_function, 
              A_ub=A_ub, 
              b_ub=b_ub, 
              bounds=X_bounds, 
              method='highs')


print("FINAL MULTI OBJECTIVE OPTIMIZATION")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

if res.success:
    for month, production_amount in zip(months, res.x):
        print(f"{month}: {round(production_amount):10,d}")
    
    print(f"\nOptimization Successful")
else:
    print("Optimization failed:", res.message)


if res.success:
    optimized_X = res.x
    
    final_sales_val = np.sum(C_sales * optimized_X)
    final_costs_val = np.sum(C_cost * optimized_X)
    final_prod_val  = np.sum(C_prod * optimized_X)
    
    print("PERFORMANCE OF THE OPTIMIZED FUNCTION")
    print(f"Total Sales:        {final_sales_val * 1000:,.0f} RON")
    print(f"Total Costs:        {final_costs_val:,.0f} RON")
    print(f"Total Productivity: {final_prod_val:,.0f} Added Value")
    

# plotting

plt.figure(figsize=(12, 7))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Plot the 4 optimization scenarios using the variables we saved
plt.plot(months, X_sales, marker='o', linestyle='-', color='blue', label='Max Turnover', linewidth=2)
plt.plot(months, X_cost, marker='s', linestyle='-', color='green', label='Min Costs', linewidth=2)
plt.plot(months, X_productivity, marker='^', linestyle='-', color='orange', label='Max Productivity', linewidth=2)
plt.plot(months, optimized_X, marker='D', linestyle='-', color='purple', label='Multi-Objective', linewidth=3)

# Plot the Physical Reality (The Capacity Bottleneck)
plt.plot(months, Cap, linestyle='--', color='red', label='FACTORY MAX CAPACITY', linewidth=2.5)

# Formatting the Y-axis to show millions (e.g., 2.5M) cleanly
def millions_formatter(x, pos):
    return f'{x / 1000000:.1f}M'

plt.gca().yaxis.set_major_formatter(FuncFormatter(millions_formatter))

plt.title('Production Schedule Optimization: Single vs. Multi-Objective', fontsize=16, fontweight='bold', pad=15)
plt.xlabel('Months', fontsize=12, fontweight='bold')
plt.ylabel('Production Quantity (Pieces)', fontsize=12, fontweight='bold')

# Grid and Legend
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper right', fontsize=10, shadow=True, facecolor='white')

# Adjust layout and display
plt.tight_layout()
plt.show()