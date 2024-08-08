import matplotlib.pyplot as plt
import matplotlib
from volatility_modelling.calibration_methods.newton_raphson_calibration import NewthonRaphson
from testing.test_data import get_test_data
import numpy as np
matplotlib.use('TkAgg')


def test_newton_raphson(initial_guess):
    data = get_test_data()
    risk_free_rate = round(data['Risk_Free_Rate'].iloc[0], 6)
    strike_prices = data["Strike_Price"]
    maturities = round(data['Time_To_Maturity'].unique()[0], 6)
    market_ivs = data["mid_iv"] / 100
    spot_price = data['Coin_Price'].iloc[0]
    expiry = data["Expiry_Date"].iloc[0]
    nr = NewthonRaphson(max_iter=100)
    implied_vols = []
    moneyness = []

    for idx, row in data.iterrows():
        k = row["Strike_Price"]
        C = row["Market_Price"]

        moneyness.append(np.log(k / spot_price))
        implied_vol = nr.nr_solution(spot_price=spot_price, strike_price=k, r=risk_free_rate, initial_guess=initial_guess, t=maturities, market_price=C)

        implied_vols.append(implied_vol)

    print(implied_vols)
    print("Expiry", data["Expiry_Date"].iloc[0])
    plt.plot(moneyness, implied_vols, label='Newton Raphson Method Vols')
    plt.plot(moneyness, market_ivs, label='Market Vols')
    plt.xlabel('Log Moneyness log(Strike/Spot)')
    plt.ylabel('Implied Volatility')
    plt.title(f"Implied Volatility Smile - {expiry}")
    plt.legend()
    plt.show()
