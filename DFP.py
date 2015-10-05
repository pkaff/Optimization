from Quasi_newton_method import *

class DFP(Quasi_newton_method):
    
    def update_matrix(self, gamma, delta):
        sum_1 = np.dot(delta*delta.T)/np.dot(delta,gamma)
        sum_2 = (self.H_k_1*np.dot(gamma,gamma.T)*h_k_1)/np.dot(gamma,self.H_k_1*gamma)
        H = self.H_k_1 + sum_1 - sum_2
        return H
