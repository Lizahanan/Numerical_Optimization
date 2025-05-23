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
    # Triangle function
    f = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)
    #gradient
    grad = np.array([np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) - np.exp(-x1 - 0.1),
                     3 * np.exp(x1 + 3*x2 - 0.1) - 3 * np.exp(x1 - 3*x2 - 0.1)])
    hess = np.array([[np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1),
                      3 * np.exp(x1 + 3*x2 - 0.1) - 3 * np.exp(x1 - 3*x2 - 0.1)],
                      [3 * np.exp(x1 + 3*x2 - 0.1) - 3 * np.exp(x1 - 3*x2 - 0.1),
                       9 * np.exp(x1 + 3*x2 - 0.1) +9 * np.exp(x1 - 3*x2 - 0.1)]]) if need_hessian else None
    return f, grad, hess