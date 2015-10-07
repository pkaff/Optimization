from Quasi_newton_method import *

class Good_broyden_method(Quasi_newton_method):

            
    def update_matrix(self, gamma, delta):
        v = gamma - np.dot(self.A_k_1,delta)
        w = delta.T
        a = 1/(np.dot(delta,delta))
        A_k = self.A_k_1 + a*np.dot(v,w)
        H_k = self.H_k_1 + (a*self.H_k_1*np.dot(v,w)*self.H_k_1)/(1-a*np.dot(w,self.H_k_1*v))
        self.A_k_1 = A_k
        return H_k
       
        
