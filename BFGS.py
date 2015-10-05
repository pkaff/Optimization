from Quasi_newton_method import *

class BFGS(Quasi_newton_method):
    
    def update_matrix(self, gamma, delta):
        sum_1 = (1+np.dot(gamma,self.H_k_1*gamma)/np.dot(delta,gamma))*(np.dot(delta,delta.T)/np.dot(delta, gamma))
        sum_2 = (np.dot(delta,gamma.T)*self.H_k_1+self.H_k_1*np.dot(gamma,delta.T))/np.dot(delta,gamma)
        H = self.H_k_1 + sum_1 - sum_2
        return H
            
