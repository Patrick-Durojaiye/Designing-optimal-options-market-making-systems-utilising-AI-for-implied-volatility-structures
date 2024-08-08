import numpy as np
from scipy.stats import norm


class BlackScholesModel:
    """
    This class implements the Black-Scholes model for option pricing.
    """

    def __init__(self):
        """
        Initializes the BlackScholesModel object.
        """
        pass

    @staticmethod
    def d1(spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        """
        Calculates the d1 component used in the Black-Scholes formula.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        r (float): The risk-free interest rate.
        sigma (float): The volatility of the underlying asset.
        t (float): The time to maturity of the option.

        Returns:
        float: The d1 value.
        """
        return (np.log(spot_price / strike_price) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))

    def d2(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        """
        Calculates the d2 component used in the Black-Scholes formula.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        r (float): The risk-free interest rate.
        sigma (float): The volatility of the underlying asset.
        t (float): The time to maturity of the option.

        Returns:
        float: The d2 value.
        """
        return self.d1(spot_price, strike_price, r, sigma, t) - sigma * np.sqrt(t)

    def call_value(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        """
        Calculates the price of a call option using the Black-Scholes formula.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        r (float): The risk-free interest rate.
        sigma (float): The volatility of the underlying asset.
        t (float): The time to maturity of the option.

        Returns:
        float: The price of the call option.
        """

        d1 = self.d1(spot_price, strike_price, r, sigma, t)
        d2 = d1 - sigma * np.sqrt(t)
        return spot_price * norm.cdf(d1) - strike_price * np.exp(-r * t) * norm.cdf(d2)

    def put_value(self, spot_price: float, strike_price: float, r: float, sigma: float, t: float):
        """
        Calculates the price of a put option using the Black-Scholes formula.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        r (float): The risk-free interest rate.
        sigma (float): The volatility of the underlying asset.
        t (float): The time to maturity of the option.

        Returns:
        float: The price of the put option.
        """

        d1 = self.d1(spot_price, strike_price, r, sigma, t)
        d2 = d1 - sigma * np.sqrt(t)
        return (strike_price * np.exp(-r * t) * norm.cdf(-d2)) - (spot_price * norm.cdf(-d1))

    def vega(self, spot_price: float, t: float, strike_price: float, r: float, sigma: float):
        """
        Calculates the Vega of an option using the Black-Scholes formula.

        Parameters:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.
        r (float): The risk-free interest rate.
        sigma (float): The volatility of the underlying asset.
        t (float): The time to maturity of the option.

        Returns:
        float: The Vega of the option.
        """

        d1 = self.d1(spot_price=spot_price, strike_price=strike_price, r=r, sigma=sigma, t=t)
        return spot_price * np.sqrt(t) * norm.pdf(d1)
