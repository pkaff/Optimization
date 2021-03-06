from Optimization_method import *

class Newton_method(Optimization_method):

    def s(self, x_k, placeholder = None):
        h = self.acc
        dim = len(x_k)
        # computing the inverse Hessian applied to the gradient 
        G = np.eye(dim)
        # computing the Hessian using finite differenzes on the gradient function
        for i in range(dim):
            unit = np.zeros(dim)
            unit[i] = 1
            G[:,i] = (self.p.grad(x_k + h*unit) - self.p.grad(x_k)) / h
        #Symmetrizing the Hessian
        G = 0.5 * (G + G.T)
        # Computing L with the cholesky method to compute the inverse Hessian
        # If G is not positive definite this will raise a LinAlgError
        try:
            L = sl.cholesky(G)
            y = sl.solve(L, self.p.grad(x_k))
            # L.T.conju() is the transposed, conjugated of L
            return -1 * sl.solve(L.T.conj(), y)
        except sl.LinAlgError:
            return -1*sl.solve(G, x_k)
