from trader.PricingModel import BlackScholes


class BisectionMethod:

    def __init__(self, max_iter: int = 1000, tolerance: float = 1e-4):
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.bs = BlackScholes()

    def bisect_solution(self, spot_price: float, strike_price: float, r: float, t: float, market_price: float,
                        lower_iv: float, upper_iv: float):

        epsilon = 1e-6

        for i in range(self.max_iter):
            mid_sigma = (lower_iv + upper_iv) / 2
            bs_price = self.bs.call_value(spot_price=spot_price, strike_price=strike_price, r=r, sigma=mid_sigma, t=t)

            error = market_price - bs_price

            print(
                f"Iteration {i}: lower_bound={lower_iv}, upper_bound={upper_iv}, mid_sigma={mid_sigma}, bs_price={bs_price}, error={error}")

            if abs(error) < epsilon:
                break

            elif error < 0:
                upper_iv = mid_sigma

            else:
                lower_iv = mid_sigma

        return mid_sigma
