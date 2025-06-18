# Numerical Optimization

Implementation of optimization algorithms for the Numerical Optimization course at Reichman University, Spring 2025.

## Overview

This repository contains implementations of both unconstrained and constrained optimization algorithms, with comprehensive testing and visualization capabilities.

## Project Structure

```
.
├── src/
│   ├── unconstrained_min.py    # Gradient descent & Newton's method
│   ├── constrained_min.py      # Interior point method (Assignment 2)
│   └── utils.py                # Plotting and visualization utilities
├── tests/
│   ├── examples.py             # Test functions (quadratic, Rosenbrock, etc.)
│   ├── test_unconstrained_min.py  # Unit tests for unconstrained optimization
│   └── test_constrained_min.py    # Unit tests for constrained optimization
├── HW1/                        # Assignment 1 implementation
└── results/                    # Generated plots and outputs
```

## Algorithms Implemented

### Assignment 1: Unconstrained Optimization
- **Gradient Descent (GD)**: Basic first-order optimization method
- **Newton's Method (NT)**: Second-order method using Hessian information
- **Backtracking Line Search**: Armijo rule for step size selection
- **Test Functions**: Circle, ellipse, rotated ellipse, Rosenbrock, linear, smooth triangle

### Assignment 2: Constrained Optimization
- **Interior Point Method**: Log-barrier method for constrained problems
- **Quadratic Programming (QP)**: Closest probability vector problem
- **Linear Programming (LP)**: Polygon vertex optimization
- **Constraint Handling**: Support for equality and inequality constraints

## Features

- **Robust Implementation**: Error handling for singular matrices and convergence issues
- **Comprehensive Testing**: Unit tests with multiple test functions
- **Rich Visualization**: 
  - 2D contour plots with optimization paths
  - 3D visualization for constrained problems
  - Function value convergence plots
- **Flexible Interface**: Configurable tolerances and iteration limits

## Installation

```bash
git clone <repository-url>
cd Numerical_Optimization
```

**Dependencies:**
- NumPy
- Matplotlib
- Python 3.7+

## Usage

### Unconstrained Optimization

```python
from src.unconstrained_min import minimize
from tests.examples import rosenbrock
import numpy as np

# Define starting point
x0 = np.array([-1.0, 2.0])

# Run optimization
result = minimize(rosenbrock, x0, method='NT', max_iter=1000)
x_final, f_final, success, x_path, f_path, info = result

print(f"Final point: {x_final}")
print(f"Final value: {f_final}")
print(f"Converged: {success}")
```

### Constrained Optimization

```python
from src.constrained_min import interior_pt
import numpy as np

# Define objective function
def quadratic_objective(x, need_hessian=False):
    # min x² + y² + (z+1)²
    f = x[0]**2 + x[1]**2 + (x[2] + 1)**2
    grad = np.array([2*x[0], 2*x[1], 2*(x[2] + 1)])
    hess = 2*np.eye(3) if need_hessian else None
    return f, grad, hess

# Define constraints
ineq_constraints = [
    lambda x, need_hessian=False: (-x[0], np.array([-1, 0, 0]), None),  # x ≥ 0
    lambda x, need_hessian=False: (-x[1], np.array([0, -1, 0]), None),  # y ≥ 0
    lambda x, need_hessian=False: (-x[2], np.array([0, 0, -1]), None)   # z ≥ 0
]

eq_constraints_mat = np.array([[1, 1, 1]])  # x + y + z = 1
eq_constraints_rhs = np.array([1])
x0 = np.array([0.1, 0.2, 0.7])

# Solve constrained problem
result = interior_pt(quadratic_objective, ineq_constraints, 
                    eq_constraints_mat, eq_constraints_rhs, x0)
```

### Running Tests

```bash
# Test unconstrained optimization
python -m tests.test_unconstrained_min

# Test constrained optimization  
python -m tests.test_constrained_min
```

## Test Problems

### Assignment 1 Problems
1. **Quadratic Functions**: Circle, ellipse, and rotated ellipse contours
2. **Rosenbrock Function**: Classic banana-shaped non-convex function
3. **Linear Function**: Simple linear objective
4. **Smooth Triangle**: Boyd's smoothed triangle function

### Assignment 2 Problems
1. **QP Problem**: Find closest probability vector to (0,0,-1)
   - Objective: min x² + y² + (z+1)²
   - Constraints: x + y + z = 1, x ≥ 0, y ≥ 0, z ≥ 0

2. **LP Problem**: Find upper right vertex of polygon  
   - Objective: max x + y
   - Constraints: y ≥ -x+1, y ≤ 1, x ≤ 2, y ≥ 0

## Algorithm Details

### Interior Point Method
- **Log-barrier approach**: Converts inequality constraints to barrier functions
- **Parameters**: Initial t = 1, increase factor μ = 10
- **Inner solver**: Uses existing unconstrained optimization methods
- **Central path**: Tracks optimization path as barrier parameter increases

### Convergence Criteria
- **Objective tolerance**: |f(x_k) - f(x_{k-1})| < 1e-12
- **Parameter tolerance**: ||x_k - x_{k-1}|| < 1e-8
- **Maximum iterations**: Configurable (default: 100)

## Visualization

The framework generates comprehensive plots:
- **Contour plots**: Show feasible regions and optimization paths
- **3D visualization**: For problems in 3D space (QP problem)
- **Convergence plots**: Objective value vs iteration number
- **Central path**: Outer iteration path for interior point method

## Results Structure

Generated plots and outputs are saved to:
```
results/
├── qp_feasible_region.png      # QP problem visualization
├── qp_objective_vs_iteration.png
├── lp_feasible_region.png      # LP problem visualization
├── lp_objective_vs_iteration.png
└── [function_name]_contour.png # Unconstrained problem plots
```

## Key Implementation Features

- **Numerical Stability**: Proper handling of singular matrices
- **Flexible Constraints**: Support for mixed equality/inequality constraints
- **Efficient Line Search**: Backtracking with Armijo condition
- **Comprehensive Logging**: Detailed iteration information
- **Error Handling**: Graceful degradation for edge cases

## Course Information

**Course**: Numerical Optimization  
**Institution**: Reichman University  
**Semester**: Spring 2025  
**Assignments**: Implementation-focused programming exercises

## License

Academic use only - for coursework at Reichman University.
