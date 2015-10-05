from Quasi_newton_method import *

class BFGS(Quasi_newton_method):
    
    def __init__(self, prob, accuracy):
        super(self.__class__, self).__init__(prob, accuracy)
        self.H_k_1 = None
        self.g_k_1 = None
    
    def s(self, x_k, x_k_1=None):
        if x_k_1 == None:
            # Initializing the Matrix and the gradient
            self.A_k_1 = np.eye(len(x_k)) 
            self.g_k_1 = self.p.grad(x_k)
            return -1* self.g_k_1
        else:
            temp = self.p.grad(x_k)
            gamma = temp - self.g_k_1
            self.g_k_1 = temp # Saving the evaluation of the gradient for the next step
            delta = x_k - x_k_1
            sum_1 = (1+np.dot(gamma,self.H_k_1*gamma)/np.dot(delta,gamma))*(np.dot(delta,delta.T)/np.dot(delta, gamma))
            sum_2 = (np.dot(delta,gamma.T)*self.H_k_1+self.H_k_1*np.dot(gamma,delta.T))/np.dot(delta,gamma)
            H = self.H_k_1 + sum_1 - sum_2
            return -1 * H*self.g_k_1