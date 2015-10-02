import unittest
from scipy import *
from numpy import *
from Newton_method import *

class TestQuasiNewton(unittest.TestCase):

    def test_newton(self):
        expected_minimum = array([1, 1])
        gradient = array([lambda x, y: 202*x - 200*y - 2, lambda x, y: 200*(y - x)])
        problem = Optimization_problem(rosen, 2, gradient)
        newton = Newton_method(problem, 0.0001)
        newton.solve(rosen)
    def rosen(x):
        return 100*((x[1] - x[0])**2) + (1 - x[0])**2

if _name__ == '__main__':
    unittest.main()
