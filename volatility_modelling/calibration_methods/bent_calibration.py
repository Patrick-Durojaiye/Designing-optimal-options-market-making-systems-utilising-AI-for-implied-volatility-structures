from scipy.optimize import brentq
from trader.PricingModel import BlackScholes


class BrentMethod:

    def __init__(self, max_iter: int = 1000, tolerance: float = 1e-4):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.bs = BlackScholes()

    def brent_solution(self, spot_price: float, strike_price: float, r: float, t: float, market_price: float):

        # Define the function whose root we want to find
        def objective_function(sigma):
            return market_price - self.bs.call_value(spot_price=spot_price, strike_price=strike_price, r=r, sigma=sigma, t=t)

        # Define initial bounds
        lower_bound = -3
        upper_bound = 3

        # Use Brent's method to find the root
        implied_volatility = brentq(objective_function, lower_bound, upper_bound, xtol=self.tolerance, maxiter=self.max_iter)

        return implied_volatility