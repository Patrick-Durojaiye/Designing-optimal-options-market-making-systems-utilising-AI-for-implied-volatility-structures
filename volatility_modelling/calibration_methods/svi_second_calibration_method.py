import numpy as np
from scipy.optimize import minimize, lsq_linear
import matplotlib
import matplotlib.pyplot as plt
from volatility_modelling.models.svi_model import SVIModel

matplotlib.use('TkAgg')


class SVICalibration:

    def __init__(self, market_strikes: np.array, spot_price: float, market_ivs: np.array, maturities: np.array):
        self.market_strikes = market_strikes
        self.spot_price = spot_price
        self.market_ivs = market_ivs
        self.maturities = maturities
        self.log_moneyness = np.log(market_strikes / spot_price)
        self.params = None
        self.svi_model = SVIModel()

    @staticmethod
    def _svi_quasi(y, a, d, c):
        return a + d * y + c * np.sqrt(np.square(y) + 1)

    def svi_quasi_rmse(self, iv, y, a, d, c):
        return np.sqrt(np.mean(np.square(self._svi_quasi(y, a, d, c) - iv)))

    def calc_adc(self, iv, m, sigma):
        y = (self.log_moneyness - m) / sigma
        s = max(sigma, 1e-6)
        bnd = ((0, 0, 0), (max(iv.max(), 1e-6), 2 * np.sqrt(2) * s, 2 * np.sqrt(2) * s))
        z = np.sqrt(np.square(y) + 1)

        A = np.column_stack([np.ones(len(iv)), np.sqrt(2) / 2 * (y + z), np.sqrt(2) / 2 * (-y + z)])

        a, d, c = lsq_linear(A, iv, bnd, tol=1e-12, verbose=False).x
        return a, np.sqrt(2) / 2 * (d - c), np.sqrt(2) / 2 * (d + c)

    def opt_msigma(self, msigma, iv):
        m, sigma = msigma
        y = (self.log_moneyness - m) / sigma
        _a, _d, _c = self.calc_adc(iv, m, sigma)
        return np.sum(np.square(_a + _d * y + _c * np.sqrt(np.square(y) + 1) - iv))

    @staticmethod
    def quasi2raw(a, d, c, m, sigma):
        # a, b, rho, m, sigma
        return a, c / sigma, d / c, m, sigma

    def quasi_calibration(self, init_msigma, maxiter=100, exit=1e-12, verbose=False):
        opt_rmse = 1
        x = self.log_moneyness
        T = self.maturities.max()
        v_tilde = (self.market_ivs ** 2 * T).flatten()

        for i in range(1, maxiter + 1):
            m_star, sigma_star = minimize(self.opt_msigma,
                                          init_msigma, v_tilde,
                                          method='Nelder-Mead',
                                          bounds=((2 * min(x.min(), 0), 2 * max(x.max(), 0)), (1e-6, 1)),
                                          tol=1e-12).x

            a_star, d_star, c_star = self.calc_adc(v_tilde, m_star, sigma_star)
            opt_rmse1 = self.svi_quasi_rmse(v_tilde, (self.log_moneyness - m_star) / sigma_star, a_star, d_star, c_star)
            if verbose:
                print(f"round {i}: RMSE={opt_rmse1} para={[a_star, d_star, c_star, m_star, sigma_star]}     ")
            if i > 1 and opt_rmse - opt_rmse1 < exit:
                break
            opt_rmse = opt_rmse1
            init_msigma = [m_star, sigma_star]

        result = np.array([a_star, d_star, c_star, m_star, sigma_star, opt_rmse1])
        if verbose:
            print(f"\nfinished. params = {result[:5].round(10)}")

        a, b, rho, m, sigma = self.quasi2raw(a=a_star, d=d_star, c=c_star, m=m_star, sigma=sigma_star)
        self.params = (a, b, rho, m, sigma)
        return self.params

    def reset_class_variable_data(self, market_strikes: np.array, spot_price: float, market_ivs: np.array, maturities: np.array):
        self.market_strikes = market_strikes
        self.spot_price = spot_price
        self.market_ivs = market_ivs
        self.maturities = maturities
        self.log_moneyness = np.log(market_strikes / spot_price)
        self.params = None

    def plot_fit(self):
        """
            Plots the volatility smile from the SVI
            :return:
            """
        if self.params is None:
            raise ValueError("Model parameters have not yet been successfully calibrated.")

        for i, T in enumerate(self.maturities):
            fitted_vols = np.sqrt([self.svi_model.evaluate_svi(self.params, lm) / T for lm in self.log_moneyness])
            plt.figure(figsize=(10, 6))
            plt.plot(self.log_moneyness, self.market_ivs[i], 'r-', label='Market Vols')
            plt.plot(self.log_moneyness, fitted_vols, 'b-', label='SVI Fit')
            plt.xlabel('Logmoneyness')
            plt.ylabel('Implied Volatility')
            plt.legend()
            plt.title(f'SVI Calibration Fit to Market Data for Maturity {T} days')
            plt.show()



