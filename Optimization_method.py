from Optimization_problem import *
from numpy import *
from scipy import *

#Abstract class Optimization_method which can be inherited from. Has a solve method which is general for all newton-methods.
class Optimization_method(object):
    #Takes a Optimization_problem object and an accuracy (stop condition: when update is smaller than accuracy)
    def __init__(self, prob, accuracy):
        self.p = prob
        self.acc = accuracy

    @p.dec
    def solve(self, f):
        self.p.fun = f
        x = x0
        while abs(x - x_pre) < accuracy:
            s = self.s(x_pre)
            a = self.alpha(x_pre, s)
            (x, x_pre) = (x_pre + a * s, x)
        return x

    def no_line_search(self):
        return 1

    def exact_line_search(self, x_k, s_k):
        return 1 #should be return argmin_alpha(x_k + alpha*s_k)

    def inexact_line_search(self, x_k, s_k):
        
