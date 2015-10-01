#defines an optimization problem from a function along with its dimensions and possibly its gradient as well as a decorator in case the user wants to solve another problem than minimization
class Optimization_problem(object):
    def __init__(self, fun, dim, grad = None, dec = lambda f: f):
        self.fun = fun
        self.dim = dim
        self.grad = grad
        self.dec = dec
