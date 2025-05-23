import numpy as np

def quad_circle(x, need_hessian=False):
    """
    Quadratic function with identity Q matrix (circle contours).
    """
    # TODO: Implement Q, f, grad, hess
    return f, grad, hess

def quad_ellipse(x, need_hessian=False):
    """
    Quadratic function with axis-aligned ellipse Q matrix.
    """
    # TODO: Implement Q, f, grad, hess
    return f, grad, hess

def quad_rotated_ellipse(x, need_hessian=False):
    """
    Quadratic function with rotated ellipse Q matrix.
    """
    # TODO: Implement Q using R.T @ D @ R, f, grad, hess
    return f, grad, hess

def rosenbrock(x, need_hessian=False):
    """
    Rosenbrock function: banana-shaped contours. Non-convex.
    """
    # TODO: Implement f, grad, hess
    return f, grad, hess

def linear(x, need_hessian=False):
    """
    Linear function f(x) = a^T x. Contour lines are straight.
    """
    # TODO: Choose vector a, implement f, grad, hess (None)
    return f, grad, None

def smooth_triangle(x, need_hessian=False):
    """
    Smoothed triangle function from Boyd's book, Ex 9.20.
    """
    # TODO: Implement f, grad, hess
    return f, grad, hess