import numpy as np
from scipy.stats import norm


class BlackScholes:

    def __init__(self):
        pass

    def d1(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        return (np.log(spot_price / strike_price) + (r + ((sigma ** 2) / 2)) * t) / sigma * np.sqrt(t)

    def d2(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        return self.d1(spot_price, strike_price, r, sigma, t) - sigma * np.sqrt(t)

    def call_value(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        return (spot_price * norm.cdf(self.d1(spot_price=spot_price, strike_price=strike_price, r=r, sigma=sigma, t=t))
                ) - (strike_price * np.exp(-r * t) * norm.cdf(self.d2(spot_price=spot_price, strike_price=strike_price,
                                                                      r=r, sigma=sigma, t=t)))

    def put_value(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        return (strike_price * np.exp(-r * t) * norm.cdf(-self.d2(spot_price=spot_price, strike_price=strike_price, r=r,
                                                                  sigma=sigma, t=t))) - (
                       spot_price * norm.cdf(-self.d1(spot_price=spot_price, strike_price=strike_price, r=r,
                                                      sigma=sigma, t=t)))
