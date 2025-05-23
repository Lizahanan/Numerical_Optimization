import numpy as np

def minimize(f, x0, method='GD', obj_tol=1e-12, param_tol=1e-8, max_iter=100):
    """
    Main minimization function.
    Parameters:
    - f: function to minimize callable, returns (f(x), grad(x), hess(x)(optional))
    - x0: initial point
    - method: optimization method, 'GD' for gradient descent or 'NT' for Newton's method
    - obj_tol: objective function tolerance set by default to 1e-12
    - param_tol: parameter tolerance set by default to 1e-8
    - max_iter: maximum number of iterations set by default to 100

    Returns:
    - x : the final location -> the minimizer
    - f_val : the final value of the objective function
    - success : boolean indicating if the algorithm converged or max_iter was reached
    - x_path : list of all points visited during the optimization process
    - f_path : list of all objective values during the optimization process
    """

    x = np.array(x0, dtype=float) #x is the initial point is a vector 
    #note that the examples implementation uses this structure
    f_val, grad, hess = f(x, need_hessian = (method == 'NT')) #f_val is the value of the objective function at x, grad is the gradient at x, hess is the Hessian at x
    x_path = [x.copy()] #x_path is a list of all points visited during the optimization process we initialize it with the initial point
    f_path = [f_val] #f_path is a list of all objective values during the optimization process we initialize it with the initial value of the objective function
    success = False #success is a boolean indicating if the algorithm converged or max_iter was reached
    last = None #last is a string that will be used to print the final iteration information


    #main iteration loop 
    for i in range(1, max_iter + 1):
        #compute the step direction
        step = _gradient_descent_step(grad) if method == 'GD' else _newton_step(grad, hess)
        #choose alpha 
        alpha = _backtracking_line_search(f, x, step, grad) 
        new_x = x + alpha * step #update the point
        #compute the new objective function value
        new_f_val, new_grad, new_hess = f(new_x, need_hessian = (method == 'NT')) #new_f_val is the value of the objective function at new_x, new_grad is the gradient at new_x, new_hess is the Hessian at new_x

        #_print_iteration_info(i, new_x, new_f_val) #print the iteration number, point, and objective value
        #check if the stopping criteria are met
        if _should_stop(f_val, new_f_val, x, new_x, obj_tol, param_tol):
            success = True #set success to True if the stopping criteria are met
            last = f"Final iteration: {i}, x = {x}, f(x) = {f_val}, successful"
            return new_x, new_f_val, success, x_path, f_path
            
        x, f_val, grad, hess = new_x, new_f_val, new_grad, new_hess #update the point, objective function value, gradient and Hessian
        x_path.append(x.copy()) #append the new point to the list of points visited
        f_path.append(f_val) #append the new objective function value to the list of objective values

    # If we reach here, it means we did not converge within max_iter
    
    last = f"Final iteration: {i}, x = {x}, f(x) = {f_val}, unsuccessful"
    
    return x, f_val, success, x_path, f_path, last #return the final point, objective function value, success status, list of points visited and list of objective values  


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
    try:
        return np.linalg.solve(hess, -grad)
    except np.linalg.LinAlgError:
        # If Hessian is singular, fall back to gradient descent
        print("Warning: hessian is singular, falling back to gradient descent.")
        return -grad


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

def _backtracking_line_search(f, x, direction, grad, alpha=1.0, rho=0.5, c=0.01):
    """
    Backtracking line search finds the step size alpha that satisfies the Wolfe conditions.
    Parameters:
    - f: function to minimize
    - x: current point
    - direction: search direction
    - grad: gradient at current point
    - alpha: initial step size
    - rho: reduction factor for step size
    - c: Wolfe condition constant
    """
    f_val, _, _ = f(x, need_hessian=False) #get the value of the objective function at x
    while True:
        #take a step 
        new_x = x + alpha * direction
        new_f_val, _, _ = f(new_x, need_hessian=False)
        #check the Wolfe condition
        if new_f_val <= f_val + c * alpha * np.dot(grad, direction):
            break
        alpha = rho * alpha
    #return the step size that satisfies the Wolfe condition
    return alpha 
    