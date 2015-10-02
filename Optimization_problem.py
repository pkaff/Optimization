from scipy import optimize as so

#defines an optimization problem from a function along with an initial guess and possibly its gradient as well as a decorator in case the user wants to solve another problem than minimization
class Optimization_problem(object):
    def __init__(self, fun, start = 0., grad = None, dec = lambda f: f):
        #if a gradient is provided by the user, it should be used. Otherwise, we use the numerical approximation of the gradient specified in the scipy.optimize package 
        if (grad != None):
            self.grad = grad
        else:
            def deriv(x):
                return so.approx_fprime(x, fun, epsilon = 1.e-8)
            self.grad = deriv
                
        self.fun = fun
        self.start = start
        self.dec = dec
