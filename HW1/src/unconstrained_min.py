import numpy as np

def minimize(f, x0, method='gradient', obj_tol=1e-6, param_tol=1e-6, max_iter=1000):
    """
    Main minimization function.
    """
    


def _gradient_descent_step(grad):
    """
    Returns the gradient descent direction: -grad
    """
    return -grad


def _newton_step(grad, hess):
    """
    Returns the Newton direction: -H⁻¹ * grad
    we should not however invert the Hessian directly
    instead we call the linear solver
    hess * pk = -grad
    """
    return np.linalg.solve(hess, -grad)


def _should_stop(prev_f, curr_f, prev_x, curr_x, obj_tol, param_tol):
    """
    Checks if termination criteria are met.
    we check for the numeric tolerance:
    1. objective tolerance -> |f(x_k) - f(x_{k-1})| < obj_tol
    2. parameter tolerance -> ||x_k - x_{k-1}|| < param_tol 
    """
    obj_change = abs(curr_f - prev_f)
    param_change = np.linalg.norm(curr_x - prev_x)
    # Check if the change in objective value is less than the tolerance
    return obj_change < obj_tol or param_change < param_tol


def _print_iteration_info(i, x, f_val):
    """
    Logs iteration number, point, and objective value.
    """
    print(f"Iter {i}: x = {x}, f(x) = {f_val}")