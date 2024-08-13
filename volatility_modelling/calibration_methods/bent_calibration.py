from trader.PricingModel import BlackScholes
from scipy.optimize import brent

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

    def brent_solution(self, spot_price: float, strike_price: float, r: float, t: float, market_price: float, moneyness: float, put_price):
        """
        Finds the implied volatility using Brent's method.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        r (float): The risk-free interest rate.
        t (float): The time to maturity of the option.
        market_price (float): The market price of the option.
        moneyness (float): The moneyness of the option (relative measure of how close the option is to being in the money).
        put_price (float): Price of put option for respective strike

        Returns:
        float: The implied volatility.
        """

        def objective_function(sigma, option_type):
            """
            Objective function to find the root for implied volatility.

            Parameters:
            sigma (float): The volatility to test.
            option_type (str): Either Put or Call, denotes which type of contract we are calibrating to

            Returns:
            float: The absolute difference between the Black-Scholes price and the market price.
            """

            if option_type == "Put":
                return abs(self.bs.put_value(spot_price=spot_price, strike_price=strike_price, r=r, sigma=sigma, t=t) - put_price)
            return abs(self.bs.call_value(spot_price=spot_price, strike_price=strike_price, r=r, sigma=sigma, t=t) - market_price)

        # Define initial bounds and option to calibrate to based on log moneyness log(K/S)
        if moneyness < 0:
            lower_bound = 0.4
            upper_bound = 3
            option_type = "Put"

        else:
            lower_bound = 0.4
            upper_bound = 3
            option_type = "Call"

        implied_volatility = brent(func=objective_function, brack=(lower_bound, upper_bound), tol=1e-7, maxiter=500, args=(option_type, ))

        return implied_volatility
