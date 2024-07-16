import numpy as np
from trader.PricingModel import BlackScholes


class NewthonRaphson:

    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.bs = BlackScholes()

    def nr_solution(self, spot_price: float, strike_price: float, r: float, initial_guess: float, t: float, market_price: float, tolerance=1e-8):

        epsilon = 1e-4
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

            new_guess = sigma - (f / f_prime)

            if abs(new_guess - sigma) < tolerance:
                return new_guess

            sigma = new_guess

        return np.nan
