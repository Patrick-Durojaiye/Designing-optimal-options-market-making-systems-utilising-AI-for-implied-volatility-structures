from trader.PricingModel import BlackScholes
from scipy.optimize import brent, minimize_scalar

class BrentMethod:
    """
    This class implements a method to calculate the implied volatility of an option
    using Brent's method for root finding.
    """

    def __init__(self, max_iter: int = 1000, tolerance: float = 1e-6):
        """
        Initializes the BrentMethod object.

        Parameters:
        max_iter (int): The maximum number of iterations for the Brent's method.
        tolerance (float): The tolerance for the root finding convergence.
        """
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.bs = BlackScholes()

    def brent_solution(self, spot_price: float, strike_price: float, r: float, t: float, market_price: float, moneyness: float):
        """
        Finds the implied volatility using Brent's method.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        r (float): The risk-free interest rate.
        t (float): The time to maturity of the option.
        market_price (float): The market price of the option.
        moneyness (float): The moneyness of the option (relative measure of how close the option is to being in the money).

        Returns:
        float: The implied volatility.
        """

        def objective_function(sigma):
            """
            Objective function to find the root for implied volatility.

            Parameters:
            sigma (float): The volatility to test.

            Returns:
            float: The absolute difference between the Black-Scholes price and the market price.
            """
            return abs(self.bs.call_value(spot_price=spot_price, strike_price=strike_price, r=r, sigma=sigma, t=t) - market_price)

        # Define initial bounds based on log moneyness log(K/S)
        if moneyness < -0.25:
            lower_bound = 0.98
            upper_bound = 20

        else:
            lower_bound = 1e-6
            upper_bound = 3

        implied_volatility = brent(func=objective_function, brack=(lower_bound, upper_bound), tol=1e-8, maxiter=50)

        return implied_volatility
