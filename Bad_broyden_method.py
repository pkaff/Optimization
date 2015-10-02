from Quasi_newton_method import *

class Bad_broyden_method(Quasi_newton_method):

    def s(self, x, x_pre = None):
        #for the initial s_0
        if x_pre == None:
            self.Q_pre = np.array(np.eye(len(x)))
            return -1 * np.array(np.eye(len(x)))
        
        delta = x - x_pre
        gamma = self.p.grad(x) - self.p.grad(x_pre)
        Q = self.Q_pre + ((gamma - self.Q_pre * delta)/delta[:,None] * delta))*delta[:,None] #delta[:, None] = delta transpose
        self.Q_pre = Q
        H = sl.inv(Q)
        return -1 * H * self.o.grad(x)
        
