from Optimization_method import *
from scipy.linalg import *

class Newton_method(Optimization_method):
    
    def alpha(x_k):
        # setting alpha to 1 as no line search is used
        return no_line_search(x_k)
    
    def s(x_k,h=1.e-5):
        # computing the inverse Hessian applied to the gradient 
        G = array(eye(Optimization_method.dim))
        # computing the Hessian using finite differenzes on the gradient function
        for i in range(p.dim):
            unit=array(zeroes(p.dim,1))
            unit[i]=1
            G[:,i]=(p.grad(x_k+h*unit)-p.grad(x_k))/h
           # fij≐(f(x1,x2,…,xi+h,…,xj+k,…,xn)−f(x1,x2,…,xi+h,…,xj,…,xn)−f(x1,x2,…,xi,…,xj+k,…,xn)+f(x1,x2,…,xi,…,xj,…,xn))/(hk)
        # Computing L with the cholesky method to compute the inverse Hessian
        # If G is not postitv definit this will raise a LinAlgError
        L = chololesky(G)
        y=solve(L,p.grad(x_k))
        # L.T.conju() is the transposed, conjugated of L
        return solve(L.T.conju(),y)
