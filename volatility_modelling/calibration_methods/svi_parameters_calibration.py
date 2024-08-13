import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from volatility_modelling.models.svi_model import SVIModel


class SVICalibration:
    """
    A class for calibrating the Stochastic Volatility Inspired (SVI) model to market data.

    This class implements the calibration of the SVI based on the Quasi Explicit method.
    """

    def __init__(self, market_strikes: np.array, spot_price: float, market_ivs: np.array, maturities: np.array):
        """
        Initialize the SVICalibration object with market data.

        :param market_strikes: Array of option strike prices
        :param spot_price: Current spot price of the underlying asset
        :param market_ivs: Array of market implied volatilities
        :param maturities: Array of option maturities
        """
        self.market_strikes = market_strikes
        self.spot_price = spot_price
        self.market_ivs = market_ivs
        self.maturities = maturities
        self.log_moneyness = np.log(market_strikes / spot_price)
        self.params = None
        self.svi_model = SVIModel()


    def linear_system_solution(self, y, v_tilde):
        """
        Solve the linear system to find SVI parameters c, d and a tilde.

        :param y: Transformed variable
        :param v_tilde: Total implied variance
        :return: Tuple of SVI parameters (c, d, a tilde)
        """
        v1 = y
        v2 = np.sqrt(y ** 2 + 1)

        A = np.array([
            [np.sum(v2 ** 2), np.sum(v1 * v2), np.sum(v2)],
            [np.sum(v1 * v2), np.sum(v1 ** 2), np.sum(v1)],
            [np.sum(v2), np.sum(v1), len(self.log_moneyness)]
        ])

        b = np.array([
            [np.sum(v_tilde * v2)],
            [np.sum(v_tilde * v1)],
            [np.sum(v_tilde)]
        ])

        c, d, a = np.linalg.solve(A, b)
        return c[0], d[0], a[0]

    def lagrange_linear_system_solution(self, y, v_tilde, lambda_val, adjustment):
        """
        Solve the linear system with Lagrange multiplier adjustment.

        :param y: Transformed variable
        :param v_tilde: Total variance
        :param lambda_val: Lagrange multiplier value
        :param adjustment: Adjustment vector for constraints
        :return: Tuple of adjusted SVI parameters (c, d, a)
        """
        adjustment = np.array(adjustment)
        v1 = y
        v2 = np.sqrt(y ** 2 + 1)

        A = np.array([
            [np.sum(v2 ** 2), np.sum(v1 * v2), np.sum(v2)],
            [np.sum(v1 * v2), np.sum(v1 ** 2), np.sum(v1)],
            [np.sum(v2), np.sum(v1), len(self.log_moneyness)]
        ])

        b = np.array([
            [np.sum(v_tilde * v2)],
            [np.sum(v_tilde * v1)],
            [np.sum(v_tilde)]
        ])

        # Apply Lagrange multiplier adjustment
        b_adjusted = b - (lambda_val * adjustment)
        c, d, a = np.linalg.solve(A, b_adjusted)

        return c[0], d[0], a[0]

    @staticmethod
    def check_constraints(c, d, a_tilde, sigma, v_tilde_max):
        """
        Check if the SVI parameters satisfy the model constraints.

        :param c: SVI parameter
        :param d: SVI parameter
        :param a_tilde: SVI parameter
        :param sigma: SVI parameter
        :param v_tilde_max: Maximum total variance
        :return: Boolean indicating whether constraints are satisfied or not
        """
        if 0 <= c <= 4 * sigma and abs(d) <= c and abs(d) <= 4 * sigma - c and 0 <= a_tilde <= v_tilde_max:
            return True
        else:
            return False

    def minimize_on_boundary(self, y, v_tilde, v_tilde_max, sigma):
        """
        Performs optimization on the boundary of the constraint set.

        :param y: Transformed variable
        :param v_tilde: Total variance
        :param v_tilde_max: Maximum total variance
        :param sigma: SVI parameter
        :return: Tuple of optimized parameters and objective function value (c, d, a tilde)
        """
        def obj_func(lambda_val, side):
            """
            Define the objective function for SVI calibration based on finding the minimum on each side of the domain D.

            :param lambda_val: Lagrange multiplier value
            :param side: Side of boundary D
            :return: Value of the objective function
            """
            c, d, a_tilde = solve_for_side(y, v_tilde, lambda_val, side)
            return np.sum(((a_tilde + d * y + c * np.sqrt(y ** 2 + 1)) - v_tilde) ** 2)

        def solve_for_side(y, v_tilde, lambda_val, side):
            """
            Solve the constrained optimization problem for a specific boundary side.

            This function applies the Lagrange multiplier method to find the optimal
            SVI parameters while satisfying the constraints on a particular boundary.

            :param y: Transformed variable
            :param v_tilde: Total variance
            :param lambda_val: Lagrange multiplier value
            :param side: Int indicating which boundary side to optimize on ranges from 0 to 1
            :return: Tuple of SVI parameters (c, d, a_tilde) for the given boundary side
            """
            adjustments = [
                [1, 0, 0],  # c = 0
                [-1, 0, 0],  # c = 4sigma
                [0, 1, 0],  # d = c
                [0, -1, 0],  # d = -c
                [1, 1, 0],  # d = 4sigma - c
                [1, -1, 0],  # d = -(4sigma - c)
                [0, 0, 1],  # a_tilde = 0
                [0, 0, -1]  # a_tilde = v_tilde_max
            ]

            c, d, a_tilde = self.lagrange_linear_system_solution(y=y, v_tilde=v_tilde, lambda_val=lambda_val, adjustment=adjustments[side])

            return c, d, a_tilde

        results = []
        for side in range(8):
            res = minimize_scalar(obj_func, args=(side,), method='brent')

            lambda_val = res.x

            c, d, a_tilde = solve_for_side(y, v_tilde, lambda_val, side)

            results.append((res.fun, c, d, a_tilde))

        return min(results, key=lambda x: x[0]) if results else None

    def objective_function(self, params, v_tilde):
        """
        Define the objective function for SVI calibration.

        :param params: SVI model parameters for m and sigma
        :param v_tilde: Total variance
        :return: Value of the objective function
        """
        m, sigma = params
        y = (self.log_moneyness - m) / sigma

        # Solve for linear system
        c, d, a_tilde = self.linear_system_solution(y=y, v_tilde=v_tilde)

        # check if within domain D
        if self.check_constraints(c=c, d=d, a_tilde=a_tilde, sigma=sigma, v_tilde_max=np.max(v_tilde)):
            obj_value = np.sum(((a_tilde + d * y + c * np.sqrt(y ** 2 + 1)) - v_tilde) ** 2)

        # if outside domain, solve constrained optimisation problem
        else:
            result = self.minimize_on_boundary(y=y, v_tilde=v_tilde, v_tilde_max=np.max(v_tilde), sigma=sigma)
            obj_value, c, d, a_tilde = result
        return obj_value

    def calibrate(self):
        """
        Perform the SVI Quasi Explicit calibration.

        :return: Tuple of calibrated SVI parameters (a, b, rho, m, sigma)
        """
        x = self.log_moneyness
        v_tilde = (self.market_ivs ** 2 * self.maturities[:, np.newaxis]).flatten()
        T = self.maturities.max()

        inital_guess = [np.mean(x), np.std(x)]

        m, sigma = minimize(self.objective_function, x0=inital_guess, args=(v_tilde,), bounds=((2*min(x), 2*max(x)),(0.0005, 1)), method="Nelder-Mead").x

        y = (x - m) / sigma

        c, d, a_tilde = self.linear_system_solution(y, v_tilde)

        if not self.check_constraints(c=c, d=d, a_tilde=a_tilde, v_tilde_max=np.max(v_tilde), sigma=sigma):
            result = self.minimize_on_boundary(y=y, v_tilde=v_tilde, v_tilde_max=np.max(v_tilde), sigma=sigma)
            if result is None:
                raise ValueError("Failed to find a valid solution on the boundary")
            _, c, d, a_tilde = result

        a = a_tilde / T
        b = c / (sigma*T)
        rho = d / c

        self.params = (a, b, rho, m, sigma)
        return self.params

    def plot_fit(self):
        """
        Plots the fitted implied volatility smile from the SVI against the marker implied volatility smile
        """
        if self.params is None:
            raise ValueError("Model parameters have not yet been successfully calibrated.")

        for i, T in enumerate(self.maturities):
            fitted_vols = np.sqrt([self.svi_model.evaluate_svi(self.params, lm) for lm in self.log_moneyness])
            plt.figure(figsize=(10, 6))
            plt.plot(self.log_moneyness, self.market_ivs[i], 'r-', label='Market Vols')
            plt.plot(self.log_moneyness, fitted_vols, 'b-', label='SVI Fit')
            plt.xlabel('Log-moneyness log(K/S)')
            plt.ylabel('Implied Volatility')
            plt.legend()
            plt.title(f'SVI Calibration Fit to Market Data for Maturity {T} days')
            plt.show()
