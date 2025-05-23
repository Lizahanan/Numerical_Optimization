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
    """
    pass


def _should_stop(prev_f, curr_f, prev_x, curr_x, obj_tol, param_tol):
    """
    Checks if termination criteria are met.
    """
    pass


def _print_iteration_info(i, x, f_val):
    """
    Logs iteration number, point, and objective value.
    """
    print(f"Iter {i}: x = {x}, f(x) = {f_val}")