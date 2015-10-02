from Quasi_newton_method import *

class Good_broyden_method(Quasi_newton_method):
    
    def __init__(self, prob, accuracy):
        super(self.__class__, self).__init__(prob, accuracy)
        self.A_k_1 = None
        self.g_k_1 = None
    
    def s(self, x_k, x_k_1=None):
        print("x_k: ", x_k, x_k_1)
        if x_k_1 == None:
            # Initializing the Matrix and the gradient
            print("hello")
            self.A_k_1 = np.eye(len(x_k)) 
            self.g_k_1 = self.p.grad(x_k)
<<<<<<< HEAD
            return -1* self.g_k_1
=======
            return -1 * self.g_k_1
>>>>>>> 5fd4c1d0eb85ac473e3497898734479a3a847e0c
        else:
            print("hello2")
            #Sherman Morrison formula
            temp = self.p.grad(x_k)
            gamma = temp - self.g_k_1
            self.g_k_1 = temp # Saving the evaluation of the gradient for the next step
            delta = x_k - x_k_1
<<<<<<< HEAD
            u = delta - self.A_k_1 * gamma
=======
            u = delta - np.dot(self.A_k_1, gamma)
>>>>>>> 5fd4c1d0eb85ac473e3497898734479a3a847e0c
            a = 1 / np.dot(u,gamma)
            A_k = self.A_k_1 - a*u*u # u[:,None] gives u transposed
            self.A_k_1 = A_k # Saving the calculated matrix for the next step
<<<<<<< HEAD
            return -1*A_k*self.g_k_1
=======
            return -1*np.dot(A_k, self.g_k_1)
>>>>>>> 5fd4c1d0eb85ac473e3497898734479a3a847e0c
            
