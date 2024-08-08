# import numpy as np
# from scipy.optimize import minimize, dual_annealing, differential_evolution
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')
# from volatility_modelling.models.svi_model import SVIModel
#
#
#
# class SVICalibration:
#
#     def __init__(self, market_strikes: np.array, spot_price: float, market_ivs: np.array, maturities: np.array):
#         self.market_strikes = market_strikes
#         self.spot_price = spot_price
#         self.market_ivs = market_ivs
#         self.maturities = maturities
#         self.log_moneyness = np.log(market_strikes / spot_price)
#         self.params = None
#         self.svi_model = SVIModel()
#         print("type log money", self.log_moneyness)
#
#     def cost_function(self, params, y, v_tidle):
#         a_tidle, d, c = params
#         return 0.5 * np.linalg.norm(a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) ** 2
#
#     def gradient_solution(self, params, y, v_tilde):
#         a, d, c = params
#         z = np.sqrt(y**2 + 1)
#         cost = a + d*y + c*z - v_tilde
#
#         grad_a = np.sum(cost)
#         grad_d = np.dot(y, cost)
#         grad_c = np.dot(z, cost)
#         return (grad_a, grad_d, grad_c)
#
#     def _calculate_adc(self, m, sigma, v_tilde):
#
#         cons = [
#             {'type': 'ineq', 'fun': lambda x: x[2]-x[1], 'jac': lambda _: (0, -1, 1)},
#             {'type': 'ineq', 'fun': lambda x: x[2]+x[1], 'jac': lambda _: (0, 1, 1)},
#             {'type': 'ineq', 'fun': lambda x: 4*sigma - x[2] - x[1], 'jac': lambda _: (0, -1, -1)},
#             {'type': 'ineq', 'fun': lambda x: x[1] + 4*sigma - x[2], 'jac': lambda _: (0, 1, -1)}
#         ]
#
#         y = (self.log_moneyness-m)/sigma
#
#         result = minimize(self.cost_function, x0=np.array([np.max(v_tilde)/2, 0, 2*sigma]), args=(y, v_tilde),
#                           method="L-BFGS-B", jac=self.gradient_solution, bounds=[(None, np.max(v_tilde)),
#                                                                                  (None, None), (0, 4*sigma)],
#                           constraints=cons)
#         return result.x, result.fun
#
#     def calibrate(self):
#
#         v_tilde = (self.market_ivs ** 2 * self.maturities[:, np.newaxis]).flatten()
#
#         # # m and sigma
#         # inital_guess = [(min(x) + max(x)) / 2, 0.1]
#         #
#         # bounds = [(2 * min(x), 2 * max(x)), (0.005, 1)]
#         result = dual_annealing(lambda x: self._calculate_adc(x[0], x[1], v_tilde)[1],
#                                 bounds=[(2*min(self.log_moneyness), 2*max(self.log_moneyness)), (1e-4, 1)],
#                                 minimizer_kwargs={"method": "nelder-mead"})
#
#         print("passed first")
#         m, sigma = result.x
#
#         a_tilde, d, c = self._calculate_adc(m=m, sigma=sigma, v_tilde=v_tilde)[0]
#
#         a = a_tilde
#         b = c / sigma
#         rho = d / c
#
#         self.params = (a, b, rho, m, sigma)
#         return self.params
#
#
#     def plot_fit(self):
#         """
#             Plots the volatility smile from the SVI
#             :return:
#         """
#         if self.params is None:
#             raise ValueError("Model parameters have not yet been successfully calibrated.")
#
#         for i, T in enumerate(self.maturities):
#             # fitted_vols = np.sqrt(self.svi(self.params[i], self.log_moneyness))
#             fitted_vols = np.sqrt([self.svi_model.evaluate_svi(self.params, lm) / T for lm in self.market_strikes])
#             plt.figure(figsize=(10, 6))
#             plt.plot(self.market_strikes, self.market_ivs[i], 'ro', label='Market Vols')
#             plt.plot(self.market_strikes, fitted_vols, 'b-', label='SVI Fit')
#             plt.xlabel('Strike Prices')
#             plt.ylabel('Implied Volatility')
#             plt.legend()
#             plt.title(f'SVI Calibration Fit to Market Data for Maturity {T} days')
#             plt.show()

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
import sys
matplotlib.use('TkAgg')
from volatility_modelling.models.svi_model import SVIModel
from scipy.optimize import minimize, dual_annealing, differential_evolution, root


# class SVICalibration:
#
#     def __init__(self, market_strikes: np.array, spot_price: float, market_ivs: np.array, maturities: np.array):
#         self.market_strikes = market_strikes
#         self.spot_price = spot_price
#         self.market_ivs = market_ivs
#         self.maturities = maturities
#         self.log_moneyness = np.log(market_strikes / spot_price)
#         self.params = None
#         self.svi_model = SVIModel()
#
#     def cost_function(self, params, y, v_tidle):
#         c, d, a_tidle = params
#         return np.sqrt(np.mean((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) ** 2))
#
#     def cost_gradient(self, params, y, v_tidle):
#         c, d, a_tidle = params
#
#         grad_c = 2 * np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) * np.sqrt(y ** 2 + 1))
#         grad_d = 2 * np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) * y)
#         grad_atilde = 2 * np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle))
#
#         return np.array([grad_c, grad_d, grad_atilde])
#
#     def linear_system_solution(self, y, v_tilde):
#         A = np.array([
#             [np.sum(np.sqrt(y**2 + 1)**2), np.sum(y*np.sqrt(y**2 + 1)), np.sum(np.sqrt(y**2 + 1))],
#             [np.sum(y * np.sqrt(y**2 + 1)), np.sum(y**2), np.sum(y)],
#             [np.sum(np.sqrt(y**2 +1)), np.sum(y), len(y)]
#         ])
#
#         b = np.array([
#             [np.sum(v_tilde * np.sqrt(y ** 2 + 1))],
#             [np.sum(v_tilde*y)],
#             [np.sum(v_tilde)]
#         ])
#
#         x = np.linalg.solve(A, b)
#         return x
#
#     @staticmethod
#     def check_domain(c, d, a_tidle, sigma, v_tilde_max):
#         return -sys.float_info.epsilon <= c <= 4*sigma and np.abs(d) <= c and np.abs(d) <= (4*sigma - c) and 0 <= \
#                a_tidle <= v_tilde_max
#
#     def gradient_solution(self, params, x, v_tilde):
#         m, sigma = params
#         y = (x - m) / sigma
#
#         linear_solution = self.linear_system_solution(y=y, v_tilde=v_tilde)
#         c = linear_solution[0][0]
#         d = linear_solution[1][0]
#         a_tilde = linear_solution[2][0]
#         v_tilde_max = np.max(v_tilde)
#         if self.check_domain(c=c, d=d, a_tidle=a_tilde, sigma=sigma, v_tilde_max=v_tilde_max):
#             return self.cost_function(params=[c, d, a_tilde], y=y, v_tidle=v_tilde)
#         else:
#             initial_guess = np.array([c, d, a_tilde])
#
#             bounds = [(-sys.float_info.epsilon, 4*sigma), (-4*sigma, 4*sigma), (0, v_tilde_max)]
#             result = minimize(self.cost_function, initial_guess, args=(y, v_tilde), jac=self.cost_gradient, bounds=bounds, method='L-BFGS-B')
#             return result.fun
#
#
#     def calibrate(self):
#
#         x = self.log_moneyness
#         v_tilde = (self.market_ivs**2 * self.maturities[:, np.newaxis]).flatten()
#         T = self.maturities.max()
#
#         # m and sigma
#         inital_guess = [(min(x) + max(x))/2, 1]
#
#         bounds = [(2*min(x), 2*max(x)), (0.005,1)]
#
#         result = dual_annealing(lambda x: self.gradient_solution, x0=inital_guess, args=(x, v_tilde),
#                                 bounds=[(2*min(self.log_moneyness), 2*max(self.log_moneyness)), (1e-4, 1)],
#                                 minimizer_kwargs={"method": "nelder-mead"})
#
#         res = minimize(self.gradient_solution, inital_guess, args=(x, v_tilde), bounds=bounds, method="Nelder-Mead")
#
#         m, sigma = res.x
#         y = (x - m) / sigma
#
#         linear_solution = self.linear_system_solution(y, v_tilde)
#         c = linear_solution[0][0]
#         d = linear_solution[1][0]
#         a_tilde = linear_solution[2][0]
#
#         if not self.check_domain(c, d, a_tilde, sigma, np.max(v_tilde)):
#             print("not in domain")
#             initial_guess = np.array([c, d, a_tilde])
#
#             constraints = [
#                 {'type': 'ineq', 'fun': lambda x: x[0]},
#                 {'type': 'ineq', 'fun': lambda x: 4 * sigma - x[0]},
#                 {'type': 'ineq', 'fun': lambda x: x[0] - abs(x[1])},
#                 {'type': 'ineq', 'fun': lambda x: 4 * sigma - x[0] - abs(x[1])},
#                 {'type': 'ineq', 'fun': lambda x: x[2]},
#                 {'type': 'ineq', 'fun': lambda x: np.max(v_tilde) - x[2]}
#             ]
#
#             result = minimize(self.cost_function, initial_guess, args=(y, v_tilde), jac=self.cost_gradient, constraints=constraints)
#             c, d, a_tilde = result.x
#
#         a = a_tilde / T
#         b = c / (sigma*T)
#         rho = d / c
#
#         self.params = (a, b, rho, m, sigma)
#         return self.params
#
#     def plot_fit(self):
#         """
#             Plots the volatility smile from the SVI
#             :return:
#         """
#         if self.params is None:
#             raise ValueError("Model parameters have not yet been successfully calibrated.")
#
#         for i, T in enumerate(self.maturities):
#             # fitted_vols = np.sqrt(self.svi(self.params[i], self.log_moneyness))
#             fitted_vols = np.sqrt([self.svi_model.evaluate_svi(self.params, lm) / T for lm in self.market_strikes])
#             plt.figure(figsize=(10, 6))
#             plt.plot(self.market_strikes, self.market_ivs[i], 'ro', label='Market Vols')
#             plt.plot(self.market_strikes, fitted_vols, 'b-', label='SVI Fit')
#             plt.xlabel('Strike Prices')
#             plt.ylabel('Implied Volatility')
#             plt.legend()
#             plt.title(f'SVI Calibration Fit to Market Data for Maturity {T} days')
#             plt.show()



class SVICalibration:

    def __init__(self, market_strikes: np.array, spot_price: float, market_ivs: np.array, maturities: np.array):
        self.market_strikes = market_strikes
        self.spot_price = spot_price
        self.market_ivs = market_ivs
        self.maturities = maturities
        self.log_moneyness = np.log(market_strikes/spot_price)
        self.params = None
        self.svi_model = SVIModel()

    @staticmethod
    def v_function(y, a, c, d):
        return a +d*y +c*np.sqrt(y**2+1)

    def cost_function(self, params, y, v_tidle):
        c, d, a_tidle = params
        return np.sum((self.v_function(y=y, a=a_tidle, c=c, d=d) - v_tidle)**2)

    def linear_system_solution(self, y, v_tilde):

        v1 = y
        v2 = np.sqrt(y**2 +1)
        A = np.array([
                [np.sum(v2**2), np.sum(v1*v2), np.sum(v2)],
                [np.sum(v1*v2), np.sum(v1**2), np.sum(v1)],
                [np.sum(v2), np.sum(v1), len(self.log_moneyness)]
            ])

        b = np.array([
                [np.sum(v_tilde * v2)],
                [np.sum(v_tilde*v1)],
                [np.sum(v_tilde)]
            ])

        c, d, a = np.linalg.solve(A, b)
        return c[0], d[0], a[0]


    def gradient_solution(self, params, x, v_tilde):
        m, sigma = params
        y = (x - m) / sigma

        c, d, a = self.linear_system_solution(y=y, v_tilde=v_tilde)

        v_tilde_max = np.max(v_tilde)

        if (0 <= c <= 4*sigma and
            abs(d) <= c and
            abs(d) <= 4*sigma - c and
            0 <= a <= v_tilde_max):
            return self.cost_function(params=[c, d, a], y=y, v_tidle=v_tilde)

        else:
            edges = [
                lambda params: (0, params[1], params[2]),
                lambda params: (4*sigma, params[1], params[2]),
                lambda params: (params[0], -params[0], params[2]),
                lambda params: (params[0], params[0], params[2]),
                lambda params: (params[0], params[1], 0),
                lambda params: (params[0], params[1], v_tilde_max)
            ]

            minimum_cost = np.inf
            inital_params = (c, d, a)
            for edge in edges:
                x0 = edge(inital_params)
                result = minimize(self.cost_function, x0=x0, args=(y, v_tilde), method='L-BFGS-B', bounds=[(0, 4*sigma), (-4*sigma, 4*sigma), (0, v_tilde_max)])

                if result.success and result.fun < minimum_cost:
                    minimum_cost = result.fun

            return minimum_cost


    def calibrate(self):

        x = self.log_moneyness
        v_tilde = (self.market_ivs**2 * self.maturities[:, np.newaxis]).flatten()
        T = self.maturities.max()

        # m and sigma
        init_sigma = max(np.std(x), 1e-4)
        inital_guess = [np.mean(x), init_sigma]

        bounds = [(2*min(x), 2*max(x)), (1e-4, np.std(x) * 10)]

        result = dual_annealing(self.gradient_solution, x0=inital_guess, args=(x, v_tilde),
                                bounds=bounds,
                                minimizer_kwargs={"method": "nelder-mead"})

        # res = minimize(self.gradient_solution, inital_guess, args=(x, v_tilde), bounds=bounds, method="Nelder-Mead")

        m, sigma = result.x
        y = (x - m) / sigma

        c, d, a = self.linear_system_solution(y, v_tilde)

        v_tilde_max = np.max(v_tilde)

        if (0 <= c <= 4 * sigma and
                abs(d) <= c and
                abs(d) <= 4 * sigma - c and
                0 <= a <= v_tilde_max):
            pass


        else:
            print("edges")
            edges = [
                lambda params: (0, params[1], params[2]),
                lambda params: (4 * sigma, params[1], params[2]),
                lambda params: (params[0], -params[0], params[2]),
                lambda params: (params[0], params[0], params[2]),
                lambda params: (params[0], params[1], 0),
                lambda params: (params[0], params[1], v_tilde_max)
            ]

            minimum_cost = np.inf
            best_params = None
            inital_params = (c, d, a)
            for edge in edges:
                x0 = edge(inital_params)
                result = minimize(self.cost_function, x0=x0, args=(y, v_tilde), method='L-BFGS-B',
                                  bounds=[(0, 4 * sigma), (-4 * sigma, 4 * sigma), (0, v_tilde_max)])

                if result.success and result.fun < minimum_cost:
                    minimum_cost = result.fun
                    best_params = result.x

            c, d, a = best_params

        a = a
        b = c / (sigma)
        rho = d / c

        # def full_svi_objective(params):
        #     a, b, rho = params
        #     v_model = a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sigma ** 2))
        #     return np.sum((v_model - v_tilde) ** 2)
        #
        # a_init = np.min(v_tilde)
        # b_init = (np.max(v_tilde) - np.min(v_tilde)) / (np.max(x) - np.min(x))
        # rho_init = 0
        #
        # bounds_full = [(0, np.max(v_tilde)), (0, np.inf), (-1, 1)]
        # result_full = minimize(full_svi_objective, [a_init, b_init, rho_init],
        #                        method='L-BFGS-B', bounds=bounds_full)
        #
        # a_opt, b_opt, rho_opt = result_full.x
        #
        # self.params = (a_opt, b_opt, rho_opt, m, sigma)
        self.params = (a, b, rho, m, sigma)
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
            plt.plot(self.log_moneyness, self.market_ivs[i], 'ro', label='Market Vols')
            plt.plot(self.log_moneyness, fitted_vols, 'b-', label='SVI Fit')
            plt.xlabel('Strike Prices')
            plt.ylabel('Implied Volatility')
            plt.legend()
            plt.title(f'SVI Calibration Fit to Market Data for Maturity {T} days')
            plt.show()