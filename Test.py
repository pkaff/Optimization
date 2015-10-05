import unittest
from Newton_method import *
from Good_broyden_method import *
from Bad_broyden_method import *
from DFP import *
from BFGS import *

#LST = 0 -> no line search
#LST = 1 -> exact line search
#LST = 2 -> Goldstein inexact line search
#LST = 3 -> WP inexact line search

class TestQuasiNewton(unittest.TestCase):

    #tests the newton method with a degree 2 polynomial in 1d
    #@unittest.skip("skip")
    def test_newton_poly2(self):
        expected_minimum = 0.0
        gradient = lambda x: 2.0 * x
        problem = Optimization_problem(self.deg2poly, np.array([2]), gradient)
        newton = Newton_method(problem, 1.e-8, 1)
        sol = newton.solve(self.deg2poly)
        np.testing.assert_array_almost_equal(sol[0], expected_minimum)

    #@unittest.skip("skip")
    def test_good_broyden_poly2(self):
        expected_minimum = 0.0
        gradient = lambda x: 2.0 * x
        problem = Optimization_problem(self.deg2poly, np.array([2]), gradient)
        broyden = Good_broyden_method(problem, 1.e-8, 1)
        sol = broyden.solve(self.deg2poly)
        np.testing.assert_array_almost_equal(sol, expected_minimum)
        
    #@unittest.skip("skip")
    def test_bad_broyden_poly2(self):
        expected_minimum = 0.0
        gradient = lambda x: 2.0 * x
        problem = Optimization_problem(self.deg2poly, np.array([2]), gradient)
        broyden = Bad_broyden_method(problem, 1.e-8, 1)
        sol = broyden.solve(self.deg2poly)
        np.testing.assert_array_almost_equal(sol[0], expected_minimum)

    def deg2poly(self, x):
        return x**2.0
    
    @unittest.skip("skip")
    def test_newton_poly3(self): #should diverge, prints overflow stuff
        expected_minimum = - 4.0/3.0 + np.sqrt(13)/3.0
        gradient = lambda x: 3.0 * x**2.0 + 8.0*x + 1
        problem = Optimization_problem(self.deg3poly, np.array([-5]), gradient)
        newton = Newton_method(problem, 1.e-8, 1)
        try:
            sol = newton.solve(self.deg3poly)
            print sol
            np.testing.assert_array_almost_equal(sol[0], expected_minimum)
        except Exception as err:
            assert('Divergence' in err.message)
            
    def deg3poly(self, x):
        return x**3.0 + 4*x**2.0 + x + 1

    #@unittest.skip("skip")
    def test_newton_poly4(self):
        expected_minimum = 0.32634546 #wolframalpha approx
        gradient = lambda x: 4.0 * x**3.0 + 15.0*x**2 + 10*x + 5
        problem = Optimization_problem(self.deg4poly, np.array([5]),gradient)
        newton = Newton_method(problem, 1.e-8, 1)
        sol = newton.solve(self.deg4poly)
        np.testing.assert_array_almost_equal(sol[0], expected_minimum)

    #@unittest.skip("skip")
    def test_good_broyden_poly4(self):
        expected_minimum = 0.32634546 #wolframalpha approx
        gradient = lambda x: 4.0 * x**3.0 + 15.0*x**2 + 10*x + 5
        problem = Optimization_problem(self.deg4poly, np.array([5]),gradient)
        broyden = Good_broyden_method(problem, 1.e-8, 1)
        sol = broyden.solve(self.deg4poly)
        np.testing.assert_array_almost_equal(sol[0], expected_minimum)
        
    def deg4poly(self, x):
        return x**4 + 5*x**3 + 5*x**2 - 5*x - 6

    #tests the newton method with the rosen function in 2d
    #DONT RUN; ITS TOO SLOW
    @unittest.skip("skip")
    def test_newton_rosen(self):
        expected_minimum = np.array([1, 1])
        gradient = lambda x: np.array([202*x[0] - 200*x[1] - 2,200*(x[1] - x[0])])
        problem = Optimization_problem(self.rosen, np.array([4, 4]), gradient)
        newton = Newton_method(problem, 1.e-8, 1)
        sol = newton.solve(self.rosen)
        np.testing.assert_array_almost_equal(sol[0], expected_minimum)

    #@unittest.skip("skip")
    def test_good_broyden_rosen(self):
        expected_minimum = np.array([1, 1])
        gradient = lambda x: np.array([202*x[0] - 200*x[1] - 2,200*(x[1] - x[0])])
        problem = Optimization_problem(self.rosen, np.array([4, 4]), gradient)
        broyden = Good_broyden_method(problem, 1.e-8, 2)
        sol = broyden.solve(self.rosen)
        np.testing.assert_array_almost_equal(sol[0], expected_minimum, 3)
        
    def rosen(self, x):
        return 100*((x[1] - x[0])**2) + (1 - x[0])**2

if __name__ == '__main__':
    unittest.main()
