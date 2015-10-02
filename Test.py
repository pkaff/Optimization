import unittest
from scipy import *
from numpy import *
from Newton_method import *

class TestQuasiNewton(unittest.TestCase):

    #tests the newton method with a degree 2 polynomial in 1d
    def test_newton_poly(self):
        expected_minimum = array([0])
        gradient = lambda x: 2.0 * x
        problem = Optimization_problem(deg2poly, 2, gradient)
        newton = Newton_method(problem, 0.0001)
        newton.solve(deg2poly)
        

    def deg2poly(x):
        return x**2.0
    #tests the newton method with the rosen function in 2d
    def test_newton_rosen(self):
        expected_minimum = array([1, 1])
        gradient = array([lambda x, y: 202*x - 200*y - 2, lambda x, y: 200*(y - x)])
        problem = Optimization_problem(rosen, array([4, 4]), gradient)
        newton = Newton_method(problem, 0.0001)
        newton.solve(rosen)

    def rosen(x):
        return 100*((x[1] - x[0])**2) + (1 - x[0])**2

if _name__ == '__main__':
    unittest.main()
