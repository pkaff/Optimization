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
        return self.exact_line_search(x_k, s_k)

    def no_line_search(self):
        return 1

    def exact_line_search(self, x_k, s_k):
        def f_alpha(a):
            return self.p.fun(x_k + a * s_k)
        #returns argmin(f(x_k + a*s_k)) with respect to a
        #temp = so.minimize(f_alpha, np.array([1.]))
        #print(temp.success)
        #return temp.x
        return so.minimize_scalar(f_alpha).x

    #search method. Takes as input x_k, s_k, some left/right conditions, upper/lower bounds for acceptible points and method parameters ro, (sigma), tau and chi.    
    def inexact_line_search(self, x_k, s_k, LC, RC, aL, aU, ro = 0.1, tau = 0.1, chi = 9., sigma = 0.7):
        a = self.p.start

        def f_a(a):
            return self.p.fun(x_k + a * s_k)

        def fp_a(a):
            return s_k*self.p.grad(x_k + a * s_k)

        while not (LC(a) and RC(a)):
            if not LC:
                B1()
            else:
                B2()
            

        def B1():
            da = extra()
            da = max(da, tau*(a - aL))
            da = min(da, chi*(a - aL))
            aL = a
            a = a + da
        
        def B2():
            aU = min(a, aU)
            temp = inter()
            temp = max(temp, aL + tau*(aU - aL))
            temp = min(temp, aU - tau*(aU - aL))
            a = temp

        def extra():
            return (a - aL)*fp_a(a)/(fp_a(aL) - fp_a(a))

        def inter():
            return ((a - aL)**2)*fp_a(aL)/(2*(f_a(aL) - f_a(a) + (a - aL)*fp_a(aL)))
            

        return [a, f_a(a)]
    
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
            return s_k*self.p.grad(x_k + a * s_k)

        def LC(a):
            return fp_a(a) >= sigma*fp_a(aL)
            
        def RC(a):
            return f_a(a) <= f_a(aL) + ro*(a - aL)*fp_a(aL)
        
        return  inexact_line_search(self, x_k, s_k, LC, RC, aL, aU, ro, tau, chi, sigma)

    #calls inexact line search for Goldstein left and right conditions
    #parameters: x_k and s_k
    #ro, tau and chi are method parameters. Accepted values for ro are in [0, 1/2]
    #aL and aU define the lower and upper bounds for acceptable points.
    def Goldstein(self, x_k, s_k, ro = 0.1, tau = 0.1, chi = 9., aL = 0, aU = 1.e99):

        assert 0 <= ro <= 0.5

        def f_a(a):
            return self.p.fun(x_k + a * s_k)

        def fp_a(a):
            return s_k*self.p.grad(x_k + a * s_k)

        def LC(a):
            return f_a(a) >= f_a(aL) + (1 - ro)*(a - aL)*fp_a(aL)
            
        def RC(a):
            return f_a(a) <= f_a(aL) + ro*(a - aL)* fp_a(aL)
        
        return  inexact_line_search(self, x_k, s_k, LC, RC, aL, aU, ro, tau, chi)
