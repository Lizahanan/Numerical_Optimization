import numpy as np
from src.unconstrained_min import minimize

def compute_barrier(x, func, ineq_constraints, t , need_hessian=True ):
    '''
    Compute the barrier function value and its gradient and Hessian.
    Parameters:
    - x: current point
    - func: objective function
    - ineq_constraints: list of inequality constraints
    - need_hessian: whether to compute the Hessian
    Returns:
    - barrier_value: value of the barrier function
    - barrier_grad: gradient of the barrier function
    - barrier_hess: Hessian of the barrier function (if needed)

    '''
    f, grad, hess = func(x, need_hessian=need_hessian)
    barier_val = 0.0
    barier_grad = np.zeros_like(grad)
    barier_hess = np.zeros((len(x), len(x))) if need_hessian else None

    #now for each constraint we compute the barrier function value, gradient and hessian
    for g in ineq_constraints:
        g_val, g_grad, g_hess = g(x, need_hessian=need_hessian)
        barier_val-=np.log(-g_val)  # barrier function value
        barier_grad -= g_grad / g_val
        if need_hessian:
            barrier_hess += (np.outer(g_grad, g_grad) / (g_val**2)) + (g_hess / -g_val)
    #now we add the objective function value, gradient and hessian
    total_val = t*f + barier_val
    total_grad = t * f + barier_grad
    total_hess = t * hess + barier_hess if need_hessian else None
    return total_val, total_grad, total_hess

def make_barrier_function(func, ineq_constraints, t):
    """Returns a barrier objective function suitable for `minimize`."""
    return lambda x, need_hessian=True: compute_barrier(x, func, ineq_constraints, t, need_hessian)



