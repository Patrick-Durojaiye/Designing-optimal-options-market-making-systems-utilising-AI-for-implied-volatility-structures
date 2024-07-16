import matplotlib.pyplot as plt
import matplotlib
from volatility_modelling.calibration_methods.bisection_calibration import BisectionMethod
from testing.test_data import get_test_data
matplotlib.use('TkAgg')


def test_bisectional_method(max_iter: int, lower_iv: float, upper_iv: float):
    data = get_test_data()
    risk_free_rate = round(data['Risk_Free_Rate'].iloc[0], 6)
    strike_prices = data["Strike_Price"]
    maturities = round(data['Time_To_Maturity'].unique()[0], 6)
    market_ivs = data["mid_iv"] / 100

    bm = BisectionMethod(max_iter=max_iter)
    implied_vols = []

    for idx, row in data.iterrows():
        k = row["Strike_Price"]
        C = row["Market_Price"]
        spot_price = row['Coin_Price']
        implied_vol = bm.bisect_solution(spot_price=spot_price, strike_price=k, r=risk_free_rate, t=maturities,
                                         market_price=C, lower_iv=lower_iv, upper_iv=upper_iv)

        implied_vols.append(implied_vol)

    print(implied_vols)
    plt.plot(strike_prices, implied_vols, label='Bisection Method Vols')
    plt.plot(strike_prices, market_ivs, label='Market Vols')
    plt.xlabel('Strike Prices')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.show()