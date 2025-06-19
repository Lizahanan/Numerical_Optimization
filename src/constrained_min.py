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
    for i, g in enumerate(ineq_constraints):
        g_val, g_grad, g_hess = g(x, need_hessian=need_hessian)
        # FEASIBILITY CHECK: g_i(x) must be < 0 (strictly)
        if g_val >= 0:
            print(f"    ERROR: Constraint {i} violated!")
            # Return infinite penalty - this point is infeasible
            return np.inf, np.full_like(grad, np.inf), None
        barrier_term = -np.log(-g_val)  # Barrier term for the constraint
        barier_val += barrier_term  
        barier_grad_term = -g_grad / g_val  # Gradient term for the constraint
        barier_grad += barier_grad_term

        #barier hessian term
        if need_hessian:
            # ∇²[-log(-g_i(x))] = (∇g_i)(∇g_i)ᵀ / g_i² - ∇²g_i / g_i
            # This comes from the product rule applied to ∇g_i / g_i
            hess_term1 = np.outer(g_grad, g_grad) / (g_val**2)  # Outer product term
            hess_term2 = -g_hess / g_val  # Direct Hessian term
            barier_hess += hess_term1 + hess_term2
    #Combine original objective with barrier terms
    total_val = t * f + barier_val
    total_grad = t * grad + barier_grad
    total_hess = t * hess + barier_hess if need_hessian else None   
    return total_val, total_grad, total_hess        
    

def make_barrier_function(func, ineq_constraints, t):
    """Returns a barrier objective function suitable for `minimize`."""
    return lambda x, need_hessian=True: compute_barrier(x, func, ineq_constraints, t, need_hessian)

def interior_point(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, mu = 10.0, tol=1e-6, max_iter=100):
    '''
    interior point method for constrained optimization
    '''
    t = 1.0  # initial barrier parameter
    x = np.array(x0, dtype=float)  # initial point
    outer_path = [x.copy()]  # path of outer iterations
    outer_f_vals=[] # path of outer function values

    for _ in range(max_iter):
        barrier_obj = make_barrier_function(func, ineq_constraints, t)
        eq_constraint = {"A": eq_constraints_mat, "b": eq_constraints_rhs}
        
        


