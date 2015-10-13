from Optimization_method import *
from abc import *

class Quasi_newton_method(Optimization_method):
    __metaclass__ = ABCMeta

    def __init__(self, prob, accuracy, lst = 1):
        super(Quasi_newton_method, self).__init__(prob, accuracy, lst)
        self.H_k_1 = None
        self.g_k_1 = None
        self.A_k_1 = None
    
    @abstractmethod
    def update_matrix(self, gamma, delta):
        pass
        
    def s(self, x_k, x_k_1=None):
        if x_k_1 == None:
            # Initializing the Matrix and the gradient
            self.H_k_1 = np.eye(len(x_k))
            self.A_k_1 = np.eye(len(x_k))
            self.g_k_1 = self.p.grad(x_k)
            return -1* self.g_k_1
        else:
            g_k = self.p.grad(x_k)
            gamma = g_k - self.g_k_1
            self.g_k_1 = g_k #save the evaluation of the gradient for the next step
            delta = x_k - x_k_1
            H_k = self.update_matrix(gamma, delta)
            self.H_k_1 = H_k
            return -1 * np.dot(H_k, g_k)            

