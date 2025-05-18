import numpy as np

class LineSearch:
    def __init__(self, method = 'gradientDescent', obj_tol=1e-12, param_tol=1e-8, max_iter=100):
        """
        Initializes the Optimizer

        Parameters:
        param method: The optimization method to use. Default is 'gradientDescent'.
        param obj_tol: tolerance for objective function change
        param param_tol: tolerance for parameter (step) change
        param max_iter: maximum number of iterations   
        """

        self.method = method 
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.path = []      #stores all x_i -> the path of the optimization
        self.f_values = []  #stores all f(x_i) 
        self.success = False #boolean -> True iff the optimization was successful
