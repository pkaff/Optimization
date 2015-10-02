from Quasi_newton_method import *

class Good_Broyden_method(Quasi_newton_method):
    
    A_k_1 = array([])
    g_k_1 = array([])
    
    def s(self, x_k, x_k_1=None):
        if x_k_1 == None:
            # Initializing the Matrix and the gradient
            A_k_1 = np.eye(len(x_k)) 
            g_k_1 = self.p.grad(x_k)
            return -1* A_k_1
        else:
            #Sherman Morrison formula
            temp = p.grad(x_k)
            gamma = temp - g_k_1
            g_k_1 = temp # Saving the evaluation of the gradient for the next step
            delta = x_k - x_k_1
            u = delta - A_k_1 * gamma
            a = 1 / np.dot(u,gamma)
            A_k = A_k_1 - a*u*u[:,None] # u[:,None] ergiebt u transponiert
            A_k_1 = A_k # Saving the calculated matrix for the next step
            return -1*A_k
            
