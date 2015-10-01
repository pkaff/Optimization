from Optimization_method import *

class Newton_method(Optimization_method):
    def alpha(x_k):
        return no_line_search(x_k)
    def s(x_k):
        G = eye(Optimization_method.dim)
        for i in range(p.dim):
            unit=zeroes(p.dim,1)
            unit[i]=1
            G[:,j]=(p.grad(x_k+h*unit)-p.grad(x_k))/h
           # fij≐(f(x1,x2,…,xi+h,…,xj+k,…,xn)−f(x1,x2,…,xi+h,…,xj,…,xn)−f(x1,x2,…,xi,…,xj+k,…,xn)+f(x1,x2,…,xi,…,xj,…,xn))/(hk)
        