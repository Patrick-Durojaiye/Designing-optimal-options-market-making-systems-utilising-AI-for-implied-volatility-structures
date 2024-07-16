import matplotlib.pyplot as plt
import matplotlib
from volatility_modelling.calibration_methods.bent_calibration import BrentMethod
from testing.test_data import get_test_data
matplotlib.use('TkAgg')


def test_brent_method():
    data = get_test_data()
    risk_free_rate = round(data['Risk_Free_Rate'].iloc[0], 6)
    strike_prices = data["Strike_Price"]
    maturities = round(data['Time_To_Maturity'].unique()[0], 6)
    market_ivs = data["mid_iv"] / 100

    bren = BrentMethod(max_iter=1000)

    implied_vols = []

    for idx, row in data.iterrows():
        k = row["Strike_Price"]
        C = row["Market_Price"]
        spot_price = row['Coin_Price']
        implied_vol = bren.brent_solution(spot_price=spot_price, strike_price=k, r=risk_free_rate, t=maturities,
                                          market_price=C)

        implied_vols.append(implied_vol)

    print(implied_vols)
    plt.plot(strike_prices, implied_vols, label='Brent Method Vols')
    plt.plot(strike_prices, market_ivs, label='Market Vols')
    plt.xlabel('Strike Prices')
    plt.ylabel('Implied Volatility')
    plt.legend()
    plt.show()
