
class Optimization_problem(object):
    def __init__(self, fun, grad = None, dec = lambda f: f):
        self.fun = fun
        self.dec = dec
