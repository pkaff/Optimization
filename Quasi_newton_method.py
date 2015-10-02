from Optimization_problem import *
import numpy as np
import scipy as sp
import scipy.linalg as sl
import scipy.spatial.distance as sd

#Abstract class Optimization_method which can be inherited from. Has a solve method which is general for all newton-methods.
class Quasi_newton_method(object):
    #Takes a Optimization_problem object and an accuracy (stop condition: when update is smaller than accuracy)
    def __init__(self, prob, accuracy):
        self.p = prob
        self.acc = accuracy
        self.solve = prob.dec(self.solve)

    def solve(self, f):
        self.p.fun = f
        x_pre = self.p.start
        s_k = self.s(x_pre)
        a = self.alpha(x_pre, s_k)
        x = x_pre + a * s_k
        while sd.euclidean(x, x_pre) > self.acc:
            s_k = self.s(x)
            a = self.alpha(x, s_k)
            x_temp = x
            x = x + a * s_k
            x_pre = x_temp
            if (sd.euclidean(x, 0) > 1.e10):
                raise Exception('Divergence')
        return x

    def no_line_search(self):
        return 1

    def exact_line_search(self, x_k, s_k):
        def f_alpha(a):
            return self.p.fun(x_k + a * s_k)
        #returns argmin(f(x_k + a*s_k)) with respect to a
        return so.minimize(f_alpha, np.array([1.])).x

    def inexact_line_search(self, x_k, s_k):
        return 1
