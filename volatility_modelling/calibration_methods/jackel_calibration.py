from py_vollib.black_scholes.implied_volatility import implied_volatility as iv

class JackelMethod:

    def __init__(self):
        pass

    @staticmethod
    def jackelsolution(spot_price: float, strike_price: float, r: float, t: float, market_price: float):

        a = iv(price=market_price, S=spot_price, K=strike_price, t=t, r=r, flag='c')
        print(a)
        return a
