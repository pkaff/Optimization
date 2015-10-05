from Quasi_newton_method import *

class Good_broyden_method(Quasi_newton_method):

    def update_matrix(self, gamma, delta):
        #Sherman Morrison formula
        u = delta - np.dot(self.H_k_1, gamma)
        a = 1 / np.dot(u,gamma)
        H_k = self.H_k_1 - a*u*u.T
        return H_k
