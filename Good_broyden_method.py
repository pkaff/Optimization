from Quasi_newton_method import *

class Good_broyden_method(Quasi_newton_method):
    
    def __init__(self, prob, accuracy):
        super(self.__class__, self).__init__(prob, accuracy)
        self.A_k_1 = np.array([])
        self.g_k_1 = np.array([])
    
    def s(self, x_k, x_k_1=None):
        if x_k_1 == None:
            # Initializing the Matrix and the gradient
            self.A_k_1 = np.eye(len(x_k)) 
            self.g_k_1 = self.p.grad(x_k)
            return -1* self.A_k_1
        else:
            #Sherman Morrison formula
            temp = self.p.grad(x_k)
            gamma = temp - self.g_k_1
            self.g_k_1 = temp # Saving the evaluation of the gradient for the next step
            delta = x_k - x_k_1
            print(delta)
            print(self.A_k_1)
            print(gamma)
            u = delta - self.A_k_1 * gamma
            print(u)
            print(gamma)
            a = 1 / np.dot(u,gamma)
            A_k = self.A_k_1 - a*[u]*[u] # u[:,None] gives u transposed
            self.A_k_1 = A_k # Saving the calculated matrix for the next step
            return -1*A_k
            