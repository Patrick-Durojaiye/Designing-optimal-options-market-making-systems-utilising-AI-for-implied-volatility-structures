import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

class SVIModel:

    def __init__(self, market_strikes: np.array, spot_price: float, market_ivs: np.array, maturities: np.array):
        """
        Initialises the SVI Model with market data

        :param market_strikes: An array of strike prices
        :param spot_price: The current spot price
        :param market_ivs: An array of market implied volatility
        :param maturities: An array of the various expires
        """
        self.market_strikes = market_strikes
        self.spot_price = spot_price
        self.market_ivs = market_ivs
        self.maturities = maturities
        self.log_moneyness = np.log(market_strikes / spot_price)
        self.params = None

    def svi(self, params, x):
        """
        Stochastic Volatility Inspired model (SVI)

        :param params: SVI paramaters
        :param x: Log moneyness
        :return:
            float : output of the SVI implied volatility
        """

        a, b, rho, m, sigma = params
        return a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sigma ** 2))

    def objective(self, params, iv):
        """
        Objective function for the SVI calibration

        :param params: SVI parameters
        :param iv: Market implied volatility
        :return:
            float : The sum of the squared difference of SVI implied volatility and market implied volatility
        """

        model_variance = self.svi(params, self.log_moneyness)
        return np.sum((model_variance - iv) ** 2)

    def calibrate(self):
        """
        Calibrate the SVI parameters for each maturity.

        This method calibrates the SVI parameters (a, b, rho, m, sigma) for each
        maturity by minimizing the difference between the SVI's implied
        volatility and the market implied volatility. The calibrated
        parameters are stored in the variable `self.params`.

        :return: List of the SVI parameters for each maturity
            - self.params (list)
        """

        # Todo: Formulate method for a less naive initial guess
        self.params = []

        for i in range(len(self.maturities)):
            iv = self.market_ivs[i]
            initial_guess = [0.1, 0.1, 0.1, 0.1, 0.1]
            bounds = [(0, np.inf), (0, np.inf), (-1, 1), (-np.inf, np.inf), (0, np.inf)]
            result = minimize(self.objective, initial_guess, args=(iv,), bounds=bounds, method='L-BFGS-B')
            self.params.append(result.x)
        return self.params

    def plot_fit(self):
        """
        Plots the volatility smile from the SVI
        :return:
        """

        for i, T in enumerate(self.maturities):
            fitted_vols = np.sqrt(self.svi(self.params[i], self.log_moneyness))
            plt.figure(figsize=(10, 6))
            plt.plot(self.log_moneyness, self.market_ivs[i], 'ro', label='Market Vols')
            plt.plot(self.log_moneyness, fitted_vols, 'b-', label='SVI Fit')
            plt.xlabel('Strike Prices')
            plt.ylabel('Implied Volatility')
            plt.legend()
            plt.title(f'SVI Calibration Fit to Market Data for Maturity {T} days')
            plt.show()

    def construct_iv_surface(self):
        """
        Constructs an implied volatility surface (IVS)

        This method constructs an IVS from the calibrated SVI parameters. For maturities that are not available the
        method interporlates the parameters.
        :raises:
            ValueError: If the SVI parameters have not yet been successfully calibrated
        :return:
            - iv_surface (np.array) : IVS data stored in an array
        """
        if self.params is None:
            raise ValueError("Model parameters have not yet been successfully calibrated.")

        iv_surface = np.zeros((len(self.maturities), len(self.market_strikes)))

        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.market_strikes):

                x = np.log(K / self.spot_price)
                if T in self.maturities:
                    idx = np.where(self.maturities == T)[0][0]
                    params = self.params[idx]

                else:
                    # If T is not in maturities, interpolation is used
                    interpolated_params = []
                    for param_set in zip(*self.params):
                        interpolated_params.append(np.interp(T, self.maturities, param_set))
                    params = interpolated_params

                iv_surface[i, j] = np.sqrt(self.svi(params, x))

        X, Y = np.meshgrid(self.log_moneyness, self.maturities)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, iv_surface, cmap='viridis')
        ax.set_xlabel('Log Moneyness')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Implied Volatility')
        plt.show()
        return iv_surface
