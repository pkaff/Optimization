import unittest
from Newton_method import *

class TestQuasiNewton(unittest.TestCase):

    #tests the newton method with a degree 2 polynomial in 1d
    def test_newton_poly2(self):
        expected_minimum = 0.0
        gradient = lambda x: 2.0 * x
        problem = Optimization_problem(self.deg2poly, np.array([2]), gradient)
        newton = Newton_method(problem, 1.e-8)
        sol = newton.solve(self.deg2poly)
        self.assertAlmostEqual(sol, expected_minimum)

    def test_good_broyden_poly2(self):
        expected_minimum = 0.0
        gradient = lambda x: 2.0 * x
        problem = Optimization_problem(self.deg2poly, np.array([2]), gradient)
        broyden = Good_Broyden_method(problem, 1.e-8)
        sol = broyden.solve(self.deg2poly)
        self.assertAlmostEqual(sol, expected_minimum)
        
    def deg2poly(self, x):
        return x**2.0
    
    def test_newton_poly3(self):
        expected_minimum = - 4/3 + np.sqrt(13)/3
        gradient = lambda x: 3.0 * x**2.0 + 8.0*x + 1
        problem = Optimization_problem(self.deg3poly, np.array([5]), gradient)
        newton = Newton_method(problem, 1.e-8)
        sol = newton.solve(self.deg3poly)
        self.assertAlmostEqual(sol, expected_minimum)

    def deg3poly(self, x):
        return x**3.0 + 4*x**2.0 + x + 1

    def test_newton_poly4(self):
        expected_minimum = 0.326345
        gradient = lambda x: 4.0 * x**3.0 + 15.0*x**2 + 10*x + 5
        problem = Optimization_problem(self.deg3poly, np.array([5]),gradient)
        newton = Newton_method(problem, 1.e-8)
        sol = newton.solve(self.deg4poly)
        self.assertAlmostEqual(sol, expected_minimum)
    def deg4poly(self, x):
        return x**4 + 5*x**3 + 5*x**2 - 5*x - 6

    #tests the newton method with the rosen function in 2d
    #def test_newton_rosen(self):
     #   expected_minimum = np.array([1, 1])
      #  gradient = lambda x: np.array([202*x[0] - 200*x[1] - 2,200*(x[1] - x[0])])
       # problem = Optimization_problem(self.rosen, np.array([4, 4]), gradient)
        #newton = Newton_method(problem, 1.e-5)
        #newton.solve(self.rosen)
        #assert(True)
        
    def rosen(self, x):
        return 100*((x[1] - x[0])**2) + (1 - x[0])**2

if __name__ == '__main__':
    unittest.main()
