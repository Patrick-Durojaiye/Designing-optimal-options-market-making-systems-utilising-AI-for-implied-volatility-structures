import numpy as np
from scipy.stats import norm


class BlackScholes:

    def __init__(self):
        pass

    def d1(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        return (np.log(spot_price / strike_price) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    def d2(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        return self.d1(spot_price, strike_price, r, sigma, t) - sigma * np.sqrt(t)

    def call_value(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        d1 = self.d1(spot_price, strike_price, r, sigma, t)
        d2 = d1 - sigma * np.sqrt(t)
        return spot_price * norm.cdf(d1) - strike_price * np.exp(-r * t) * norm.cdf(d2)

    def put_value(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        d1 = self.d1(spot_price, strike_price, r, sigma, t)
        d2 = d1 - sigma * np.sqrt(t)
        return (strike_price * np.exp(-r * t) * norm.cdf(-d2)) - (spot_price * norm.cdf(-d1))

    def vega(self, spot_price: float, t: float, strike_price: float, r: float, sigma: float):
        d1 = self.d1(spot_price=spot_price, strike_price=strike_price, r=r, sigma=sigma, t=t)
        return spot_price * np.sqrt(t) * norm.pdf(d1)
