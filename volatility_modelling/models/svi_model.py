import numpy as np

class SVIModel:
    """
    This class implements the Stochastic Volatility Inspired (SVI) model
    for option pricing.
    """

    def __init__(self):
        pass

    @staticmethod
    def evaluate_svi(params, x):
        """
        Evaluates the SVI (Stochastic Volatility Inspired) function.

        Parameters:
        params (tuple): A tuple containing the parameters (a, b, rho, m, sigma) of the SVI model.
        x (float): The log moneyness of an option contract defined by log(K/S)

        Returns:
        float: The value of the SVI function.
        """
        a, b, rho, m, sigma = params
        svi_value = a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))

        return svi_value
