from black_scholes import BlackScholesModel
import numpy as np
from scipy.stats import norm


class Greeks:

    def __init__(self):
        self.bsm = BlackScholesModel

    def vega(self, spot_price: float, t: float, strike_price: float, r: float, sigma: float):
        vega_value = spot_price * np.sqrt(t) * norm.pdf(self.bsm.d1(spot_price=spot_price, strike_price=strike_price,
                                                                    r=r, sigma=sigma, t=t))
        return vega_value
