from Quasi_newton_method import *

class Good_broyden_method(Quasi_newton_method):

    def update_matrix(self, gamma, delta):
        H_k = self.H_k_1 + ((gamma - self.H_k_1 * delta)/(delta.T * delta))*delta.T
        try:
            L = sl.cholesky(H_k)
            y = sl.solve(L, np.ones(len(self.g_k_1))) #just the inverse
            return sl.solve(L.T.conj(), y)
        except LinAlgError:
            H = sl.inv(H_k)
            return H
       
        
