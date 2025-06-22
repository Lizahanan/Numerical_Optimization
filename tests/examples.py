from math import sqrt
import numpy as np

def quad_circle(x, need_hessian=False):
    """
    Quadratic function with identity Q matrix (circle contours).
    """
    Q = np.eye(2) # Identity matrix for circle -> 2 x 2
    return compute_Quads(x, Q, need_hessian)

def quad_ellipse(x, need_hessian=False):
    """
    Quadratic function with axis-aligned ellipse Q matrix.
    """
    Q = np.array([[1, 0], [0, 100]]) # Ellipse matrix
    return compute_Quads(x, Q, need_hessian)

def quad_rotated_ellipse(x, need_hessian=False):
    """
    Quadratic function with rotated ellipse Q matrix.
    """
    R = np.array([[sqrt(3)/2, -0.5], [0.5, sqrt(3)/2]]) # Rotation matrix
    Q = R.T @ np.array([[100, 0], [0, 1]]) @ R # Ellipse matrix
    return compute_Quads(x, Q, need_hessian)

def compute_Quads(x, Q, need_hessian):
    '''Computes the quadratic function, gradient and hessian(if needed) for a given Q matrix'''
    #the function is of the form x^T Q x
    f = x.T @ Q @ x
    #gradient is of the form 2 Q x
    grad = 2 * Q @ x
    #hessian is constant and equal to 2 Q
    hess = 2 * Q if need_hessian else None
    return f, grad, hess

def rosenbrock(x, need_hessian=False):
    """
    Rosenbrock function: banana-shaped contours. Non-convex.
    Note that we have computed gradient and hessian for it in dry HW2
    """
    # x is a two dimensional vector
    x1, x2 = x[0], x[1]
    # Rosenbrock function
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    # Gradient
    grad = np.array([-400 * x1 * (x2 -x1**2) + 2 * x1 - 2, 
                     200 * (x2 - x1**2)])
    # Hessian
    hess = np.array([[1200 * x1**2 - 400 *x2 + 2, -400 * x1],
                     [-400 * x1, 200]]) if need_hessian else None

    return f, grad, hess

def linear(x, need_hessian=False):
    """
    Linear function f(x) = a^T x. Contour lines are straight.
    """
    a = np.array([1.0, 123.0]) # Linear function coefficients
    # Linear function
    f = a.T @ x
    # Gradient is constant and equal to a
    grad = a
    return f, grad, None

def smooth_triangle(x, need_hessian=False):
    """
    Smoothed triangle function from Boyd's book, Ex 9.20.
    """
    # x is a two dimensional vector
    x1, x2 = x[0], x[1]
    exp1 = np.exp(x1 + 3*x2 - 0.1)
    exp2 = np.exp(x1 - 3*x2 - 0.1)
    exp3 = np.exp(-x1 - 0.1)
    # Triangle function
    f = exp1 + exp2 + exp3
    #gradient
    grad = np.array([exp1 + exp2 - exp3,
                     3 * exp1 - 3 * exp2])
    hess = np.array([[exp1 + exp2 + exp3,
                      3 * exp1 - 3 * exp2],
                      [3 * exp1 - 3 * exp2,
                       9 * exp1 +9 * exp2]]) if need_hessian else None
    return f, grad, hess

def qp_objective(x, need_hessian=False):
    '''The objective function for the quadratic programming problem'''
    #x = [x,y,z]
    f = x[0]**2 + x[1]**2 + (x[2]+1)**2
    grad = np.array([2*x[0], 2*x[1], 2*(x[2]+1)])
    # Hessian is constant and equal to 2 * I
    hess = 2 * np.eye(3) if need_hessian else None

    return f, grad, hess

def qp_ineq_constraint_x(x, need_hessian=False):
    '''Inequality constraints for the quadratic programming problem
    x >= 0 (-x <= 0)
    Interior point method requires g(x) < 0, so g(x) = -x
    '''
    g_val = -x[0]  # g(x) = -x
    g_grad = np.array([-1, 0, 0])  # Gradient is constant
    g_hess = np.zeros((3, 3)) if need_hessian else None  # Hessian is zero

    return g_val, g_grad, g_hess

def qp_ineq_constraint_y(x, need_hessian=False):
    '''Inequality constraints for the quadratic programming problem
    y >= 0 (-y <= 0)
    Interior point method requires g(x) < 0, so g(x) = -y
    '''
    g_val = -x[1]  # g(x) = -y
    g_grad = np.array([0, -1, 0])  # Gradient is constant
    g_hess = np.zeros((3, 3)) if need_hessian else None  # Hessian is zero

    return g_val, g_grad, g_hess

def qp_ineq_constraint_z(x, need_hessian=False):
    '''Inequality constraints for the quadratic programming problem
    z >= 0 (-z <= 0)
    Interior point method requires g(x) < 0, so g(x) = -z 
    '''
    g_val = -x[2]   # g(x) = -z 
    g_grad = np.array([0, 0, -1])  # Gradient is constant
    g_hess = np.zeros((3, 3)) if need_hessian else None  # Hessian is zero

    return g_val, g_grad, g_hess

# QP constraint matrices for equality constraint: x + y + z = 1
qp_eq_constraints_mat = np.array([[1.0, 1.0, 1.0]])  # A matrix: [1, 1, 1]
qp_eq_constraints_rhs = np.array([1.0])              # b vector: [1]

# QP inequality constraints list
qp_ineq_constraints = [
    qp_ineq_constraint_x,
    qp_ineq_constraint_y,
    qp_ineq_constraint_z
]

qp_x0 = np.array([0.1, 0.2, 0.7])  # Initial point for QP problem

def lp_objective(x, need_hessian=False):
    '''
    Objective function for LP example: -x - y (since we're minimizing)
    Original problem: max[x + y], converted to min[-x - y]
    '''
    # x = [-x-y]
    f=-x[0] - x[1]  # Objective function: -x - y

    grad = np.array([-1, -1])  # Gradient is constant: [-1, -1]

    hess = np.zeros((2, 2)) if need_hessian else None  # Hessian is zero

    return f, grad, hess

def lp_ineq_constraint1(x, need_hessian=False):
    '''
    y >= -x + 1
    Interior point method requires g(x) < 0, so g(x) = -x - y + 1
    '''
    g_val = -x[0] - x[1] + 1  # g(x) = -x - y + 1
    g_grad = np.array([-1, -1])  # Gradient is constant: [-1, -1]
    g_hess = np.zeros((2, 2)) if need_hessian else None  # Hessian is zero

    return g_val, g_grad, g_hess

def lp_ineq_constraint2(x, need_hessian=False):
    '''
    y<= 1 
    g(x) = y - 1
    '''
    g_val = x[1] - 1  # g(x) = y - 1
    g_grad = np.array([0, 1])  # Gradient is constant: [0, 1]
    g_hess = np.zeros((2, 2)) if need_hessian else None  # Hessian is zero

    return g_val, g_grad, g_hess

def lp_ineq_constraint3(x, need_hessian=False):
    '''
    x <= 2
    g(x) = x - 2
    '''
    g_val = x[0] - 2  # g(x) = x -2
    g_grad = np.array([1, 0])  # Gradient is constant: [-1, 0]
    g_hess = np.zeros((2, 2)) if need_hessian else None  # Hessian is zero

    return g_val, g_grad, g_hess

def lp_ineq_constraint4(x, need_hessian=False):
    '''
    y >= 0
    g(x) = -x
    '''
    g_val = -x[1]  # g(x) = -y
    g_grad = np.array([0, -1])  # Gradient is constant: [-1, 0]
    g_hess = np.zeros((2, 2)) if need_hessian else None  # Hessian is zero

    return g_val, g_grad, g_hess

# LP inequality constraints list
lp_ineq_constraints = [
    lp_ineq_constraint1,  # y ≥ -x + 1
    lp_ineq_constraint2,  # y ≤ 1
    lp_ineq_constraint3,  # x ≤ 2
    lp_ineq_constraint4   # y ≥ 0
]

# LP has no equality constraints
lp_eq_constraints_mat = None
lp_eq_constraints_rhs = None

lp_x0 = np.array([0.5, 0.75])  # Initial point for LP problem
