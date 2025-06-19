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

# ================================================================
# PART 3: MAIN ALGORITHM - THE INTERIOR POINT METHOD
# ================================================================

def interior_point(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, mu = 10.0, tol=1e-6, max_iter=100):
    '''
    This is the main algorithm that implements the interior point method.
    
    ALGORITHM OVERVIEW:
    ------------------
    1. Start with t = 1 (small, so barriers dominate)
    2. Solve: minimize t*f(x) + Σ(-log(-g_i(x)))
    3. Increase t by factor μ (makes original objective more important)
    4. Repeat until convergence
    
    As t increases, the solution path traces the "central path" toward
    the optimal solution of the original constrained problem.
    '''

    # Initialize parameters
    t = 1.0  # initial barrier parameter
    x = np.array(x0, dtype=float)  # initial point
    outer_path = [x.copy()]  # path of outer iterations
    outer_f_vals=[] # path of outer function values

    #feasibility check 
    if not _check_feasibility(x, ineq_constraints, eq_constraints_mat, eq_constraints_rhs):
        raise ValueError("Initial point is not feasible! Must satisfy all constraints strictly.")
    print("✓ Initial point is feasible")

    success = False  # convergence flag

    #main outer loop
    for outer_iter in range(max_iter):

        #create the barier function for current t 
        barrier_obj= make_barrier_function(func, ineq_constraints, t)

        #handle equality constraints using quadratic penalty method
        if eq_constraints_mat is not None and eq_constraints_rhs is not None:
            barrier_obj = _add_equality_penalty(barrier_obj, eq_constraints_mat, eq_constraints_rhs)
        
        #solve the barrier problem using the minimize function
        try:
            result = minimize(
                barrier_obj,        # The barrier function (now unconstrained!)
                x,                  # Starting point for this subproblem
                method='NT',        # Newton's method (best for barrier functions)
                max_iter=1000,      # Allow many inner iterations
                obj_tol=1e-10,      # Very tight tolerance for accuracy
                param_tol=1e-8
            )
            
            x_new, barrier_val, inner_success, inner_path, inner_f_vals, inner_info = result
            
            if inner_success:
                print(f"✓ Inner optimization converged in {len(inner_f_vals)} iterations")
            else:
                print(f"⚠ Inner optimization did not converge")
            
        except Exception as e:
            print(f"✗ Inner optimization failed: {e}")
            break 

        #evaluate the objective function at the new point
        f_val, _, _ = func(x_new, need_hessian=False)

        outer_path.append(x_new.copy())
        outer_f_vals.append(f_val)

        #check if the stopping criteria are met
        m = len(ineq_constraints)

        #criteria 1: duality gap
        if m>0:
            duality_gap = m/t
            if duality_gap < tol:
                success = True
                print(f"✓ Converged: Duality gap {duality_gap:.2e} < {tol}")
                break

        #criteria 2: x parameter change
        if outer_iter >0:
            param_change = np.linalg.norm(x_new - x)
            if param_change < tol:
                success = True
                print(f"✓ Converged: Parameter change {param_change:.2e} < {tol}")
                break
        
        #update for next iteration
        x = x_new
        t *= mu
    
    # Final results
    final_f_val, _, _ = func(x, need_hessian=False)
    return x, final_f_val, success, outer_path, outer_f_vals

# ========================================================
# Helper function to check feasibility of the initial point
# ========================================================



        
        


