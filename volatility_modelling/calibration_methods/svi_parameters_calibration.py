import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
import sys
matplotlib.use('TkAgg')
from volatility_modelling.models.svi_model import SVIModel

class SVICalibration:

    def __init__(self, market_strikes: np.array, spot_price: float, market_ivs: np.array, maturities: np.array):
        self.market_strikes = market_strikes
        self.spot_price = spot_price
        self.market_ivs = market_ivs
        self.maturities = maturities
        self.log_moneyness = np.log(market_strikes / spot_price)
        self.params = None
        self.svi_model = SVIModel()

    def cost_function(self, params, y, v_tidle):
        c, d, a_tidle = params
        return np.sqrt(np.mean((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) ** 2))

    def cost_gradient(self, params, y, v_tidle):
        c, d, a_tidle = params

        grad_c = 2 * np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) * np.sqrt(y ** 2 + 1))
        grad_d = 2 * np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) * y)
        grad_atilde = 2 * np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle))

        return np.array([grad_c, grad_d, grad_atilde])

    def linear_system_solution(self, y, v_tilde):
        A = np.array([
            [np.sum(np.sqrt(y**2 + 1)**2), np.sum(y*np.sqrt(y**2 + 1)), np.sum(np.sqrt(y**2 + 1))],
            [np.sum(y * np.sqrt(y**2 + 1)), np.sum(y**2), np.sum(y)],
            [np.sum(np.sqrt(y**2 +1)), np.sum(y), len(y)]
        ])

        b = np.array([
            [np.sum(v_tilde * np.sqrt(y ** 2 + 1))],
            [np.sum(v_tilde*y)],
            [np.sum(v_tilde)]
        ])

        x = np.linalg.solve(A, b)
        return x

    @staticmethod
    def check_domain(c, d, a_tidle, sigma, v_tilde_max):
        return -sys.float_info.epsilon <= c <= 4*sigma and np.abs(d) <= c and np.abs(d) <= (4*sigma - c) and 0 <= \
               a_tidle <= v_tilde_max

    def gradient_solution(self, params, x, v_tilde):
        m, sigma = params
        y = (x - m) / sigma

        linear_solution = self.linear_system_solution(y=y, v_tilde=v_tilde)
        c = linear_solution[0][0]
        d = linear_solution[1][0]
        a_tilde = linear_solution[2][0]
        v_tilde_max = np.max(v_tilde)
        if self.check_domain(c=c, d=d, a_tidle=a_tilde, sigma=sigma, v_tilde_max=v_tilde_max):
            return self.cost_function(params=[c, d, a_tilde], y=y, v_tidle=v_tilde)
        else:
            initial_guess = np.array([c, d, a_tilde])

            bounds = [(-sys.float_info.epsilon, 4*sigma), (-4*sigma, 4*sigma), (0, v_tilde_max)]
            result = minimize(self.cost_function, initial_guess, args=(y, v_tilde), jac=self.cost_gradient, bounds=bounds, method='L-BFGS-B')
            return result.fun


    def calibrate(self):

        x = self.log_moneyness
        v_tilde = (self.market_ivs**2 * self.maturities[:, np.newaxis]).flatten()
        T = self.maturities.max()

        # m and sigma
        inital_guess = [(min(x) + max(x))/2, 1]

        bounds = [(2*min(x), 2*max(x)), (0.005,1)]
        res = minimize(self.gradient_solution, inital_guess, args=(x, v_tilde), bounds=bounds, method="Nelder-Mead")

        m, sigma = res.x
        y = (x - m) / sigma

        linear_solution = self.linear_system_solution(y, v_tilde)
        c = linear_solution[0][0]
        d = linear_solution[1][0]
        a_tilde = linear_solution[2][0]

        if not self.check_domain(c, d, a_tilde, sigma, np.max(v_tilde)):
            print("not in domain")
            initial_guess = np.array([c, d, a_tilde])

            constraints = [
                {'type': 'ineq', 'fun': lambda x: x[0]},
                {'type': 'ineq', 'fun': lambda x: 4 * sigma - x[0]},
                {'type': 'ineq', 'fun': lambda x: x[0] - abs(x[1])},
                {'type': 'ineq', 'fun': lambda x: 4 * sigma - x[0] - abs(x[1])},
                {'type': 'ineq', 'fun': lambda x: x[2]},
                {'type': 'ineq', 'fun': lambda x: np.max(v_tilde) - x[2]}
            ]

            result = minimize(self.cost_function, initial_guess, args=(y, v_tilde), jac=self.cost_gradient, constraints=constraints)
            c, d, a_tilde = result.x

        a = a_tilde / T
        b = c / (sigma*T)
        rho = d / c

        self.params = (a, b, rho, sigma, m)
        return self.params

    def plot_fit(self):
        """
            Plots the volatility smile from the SVI
            :return:
        """
        if self.params is None:
            raise ValueError("Model parameters have not yet been successfully calibrated.")

        for i, T in enumerate(self.maturities):
            # fitted_vols = np.sqrt(self.svi(self.params[i], self.log_moneyness))
            fitted_vols = np.sqrt([self.svi_model.evaluate_svi(self.params, lm) / T for lm in self.market_strikes])
            plt.figure(figsize=(10, 6))
            plt.plot(self.market_strikes, self.market_ivs[i], 'ro', label='Market Vols')
            plt.plot(self.market_strikes, fitted_vols, 'b-', label='SVI Fit')
            plt.xlabel('Strike Prices')
            plt.ylabel('Implied Volatility')
            plt.legend()
            plt.title(f'SVI Calibration Fit to Market Data for Maturity {T} days')
            plt.show()
