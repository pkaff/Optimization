from Quasi_newton_method import *

class Bad_broyden_method(Quasi_newton_method):

    def __init__(self, prob, accuracy):
        super(self.__class__, self).__init__(prob, accuracy)
        self.Q_pre = None

    def s(self, x, x_pre = None):
        #for the initial s_0
        if x_pre == None:
            self.Q_pre = np.array(np.eye(len(x)))
            return -1 * np.array(np.eye(len(x)))
        
        delta = x - x_pre
        gamma = self.p.grad(x) - self.p.grad(x_pre)
        #delta[:, None] = delta transpose
        Q = self.Q_pre + ((gamma - self.Q_pre * delta)/(delta.T * delta))*delta.T
        self.Q_pre = Q
        try:
            L = sl.cholesky(Q)
            y = sl.solve(L, self.p.grad(x))
            return -1 * sl.solve(L.T.conj(), y)
        except LinAlgError:
            H = sl.inv(Q)
            return -1 * H * self.p.grad(x)
        
