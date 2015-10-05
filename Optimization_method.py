from Optimization_problem import *
import numpy as np
import scipy as sp
import scipy.linalg as sl
import scipy.spatial.distance as sd

#Abstract class Optimization_method which can be inherited from. Has a solve method which is general for all newton-methods.
class Optimization_method(object):
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
        if (sd.euclidean(x, 0) > 1.e10):
            raise Exception('Divergence')
        while sd.euclidean(x, x_pre) > self.acc:
            s_k = self.s(x, x_pre)
            a = self.alpha(x, s_k)
            x_temp = x
            x = x + a * s_k
            x_pre = x_temp
            if (sd.euclidean(x, 0) > 1.e10):
                raise Exception('Divergence')
        return x

    def alpha(self, x_k, s_k):
        #return self.no_line_search()
        #return self.exact_line_search(x_k, s_k)
        #return self.WP(x_k, s_k) #inexact line search
        return self.Goldstein(x_k, s_k) #inexact line search 

    def no_line_search(self):
        return 1

    def exact_line_search(self, x_k, s_k):
        def f_alpha(a):
            return self.p.fun(x_k + a * s_k)
        #returns argmin(f(x_k + a*s_k)) with respect to a
        return so.minimize_scalar(f_alpha).x

    #search method. Takes as input x_k, s_k, some left/right conditions, upper/lower bounds for acceptible points and method parameters ro, (sigma), tau and chi.    
    def inexact_line_search(self, x_k, s_k, LC, RC, aL, aU, ro = 0.1, tau = 0.1, chi = 9., sigma = 0.7):

        a0 = 0.1

        def f_a(a):
            return self.p.fun(x_k + a * s_k)
            
        def fp_a(a):
            return np.dot(s_k, self.p.grad(x_k + a * s_k))         

        def extra(a0, aL):
            return (a0 - aL)*fp_a(a0)/(fp_a(aL) - fp_a(a0))

        def inter(a0, aL):
            return ((a0 - aL)**2)*fp_a(aL)/(2*(f_a(aL) - f_a(a0) + (a0 - aL)*fp_a(aL)))   

        def B1(a0, aL):
            da = extra(a0, aL)
            da = max(da, tau*(a0 - aL))
            da = min(da, chi*(a0 - aL))
            return [a0 + da, a0]
        
        def B2(a0, aL, aU):
            aU = min(a0, aU)
            temp = inter(a0, aL)
            temp = max(temp, aL + tau*(aU - aL))
            temp = min(temp, aU - tau*(aU - aL))
            return [temp, aU]
            
        while not (LC(a0) and RC(a0)):
            if not LC(a0):
                [a0, aL] = B1(a0, aL)
            else:
                [a0, aU] = B2(a0, aL, aU)

        return a0
    
    #calls inexact line search for Wolfe-Powell left and right conditions
    #parameters: x_k and s_k
    #ro, sigma, tau and chi are method parameters. Accepted values for ro and sigma are ro in [0, 1/2] and sigma in (ro, 1]
    #aL and aU define the lower and upper bounds for acceptable points.
    def WP(self, x_k, s_k, ro = 0.1, sigma = 0.7, tau = 0.1, chi = 9., aL = 0, aU = 1.e99):

        assert 0 <= ro <= 0.5
        assert ro < sigma <= 1

        def f_a(a):
            return self.p.fun(x_k + a * s_k)

        def fp_a(a):
            return np.dot(s_k, self.p.grad(x_k + a * s_k))

        def LC(a):
            return fp_a(a) >= sigma*fp_a(aL)
            
        def RC(a):
            return f_a(a) <= (f_a(aL) + ro*(a - aL)*fp_a(aL))
        
        return self.inexact_line_search(x_k, s_k, LC, RC, aL, aU, ro, tau, chi, sigma)

    #calls inexact line search for Goldstein left and right conditions
    #parameters: x_k and s_k
    #ro, tau and chi are method parameters. Accepted values for ro are in [0, 1/2]
    #aL and aU define the lower and upper bounds for acceptable points.
    def Goldstein(self, x_k, s_k, ro = 0.1, tau = 0.1, chi = 9., aL = 0, aU = 1.e99):

        assert 0 <= ro <= 0.5

        def f_a(a):
            return self.p.fun(x_k + a * s_k)

        def fp_a(a):
            return np.dot(s_k, self.p.grad(x_k + a * s_k))

        def LC(a):
            return f_a(a) >= (f_a(aL) + (1 - ro)*(a - aL)*fp_a(aL))
            
        def RC(a):
            return f_a(a) <= (f_a(aL) + ro*(a - aL)* fp_a(aL))
        
        return self.inexact_line_search(x_k, s_k, LC, RC, aL, aU, ro, tau, chi)
