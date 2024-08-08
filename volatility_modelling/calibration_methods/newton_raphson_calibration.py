import numpy as np
from trader.PricingModel import BlackScholes


class NewthonRaphson:
    """
    This class implements the Newton-Raphson method for finding the implied volatility
    of an option using the Black-Scholes model.
    """

    def __init__(self, max_iter: int):
        """
        Initializes the NewtonRaphson object.

        Parameters:
        max_iter (int): The maximum number of iterations for the Newton-Raphson method.
        """

        self.max_iter = max_iter
        self.bs = BlackScholes()

    def nr_solution(self, spot_price: float, strike_price: float, r: float, initial_guess: float, t: float, market_price: float, tolerance=1e-8):
        """
        Finds the implied volatility using the Newton-Raphson method.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        r (float): The risk-free interest rate.
        initial_guess (float): The initial guess for the volatility.
        t (float): The time to maturity of the option.
        market_price (float): The market price of the option.
        tolerance (float): The tolerance for the root finding convergence. Default is 1e-8.

        Returns:
        float: The implied volatility. Returns NaN if the solution does not converge.
        """

        epsilon = 1e-4 # Small value to check for near-zero Vega
        sigma = initial_guess

        for i in range(self.max_iter+1):

            bs_price = self.bs.call_value(spot_price=spot_price, strike_price=strike_price, r=r, sigma=sigma, t=t)
            bs_price = round(bs_price, 8)
            f = bs_price - market_price
            f_prime = self.bs.vega(spot_price=spot_price, t=t, strike_price=strike_price, r=r, sigma=sigma)

            if abs(f_prime) < epsilon:
                print("Vega is too small, breaking out of the loop.")
                print("Vega is", f_prime)
                break

            # Update the guess for sigma
            new_guess = sigma - (f / f_prime)

            # Check for convergence
            if abs(new_guess - sigma) < tolerance:
                return new_guess

            sigma = new_guess

        return np.nan
