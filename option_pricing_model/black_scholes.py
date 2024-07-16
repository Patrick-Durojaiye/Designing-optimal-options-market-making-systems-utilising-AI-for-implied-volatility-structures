import numpy as np
from scipy.stats import norm


class BlackScholesModel:

    def __init__(self):
        pass

    @staticmethod
    def d1(spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        return (np.log(spot_price / strike_price) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    def d2(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        return self.d1(spot_price, strike_price, r, sigma, t) - sigma * np.sqrt(t)

    def call_value(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        call_price = spot_price * norm.cdf(self.d1(spot_price, strike_price, r, sigma, t)) - strike_price * np.exp(
            -r * t) \
                     * norm.cdf(self.d2(spot_price, strike_price, r, sigma, t))
        return call_price

    def put_value(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        put_price = (strike_price * np.exp(-r * t) * norm.cdf(-self.d2(spot_price, strike_price, r, sigma, t))) - \
                    (spot_price * norm.cdf(-self.d1(spot_price, strike_price, r, sigma, t)))
        return put_price
