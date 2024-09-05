import matplotlib.pyplot as plt
import matplotlib
from volatility_modelling.calibration_methods.bent_calibration import BrentMethod
from testing.test_data import get_test_data
import numpy as np
matplotlib.use('TkAgg')
from datetime import datetime


def test_brent_method():
    data = get_test_data()
    risk_free_rate = round(data['Risk_Free_Rate'].iloc[0], 6)
    maturities = round(data['Time_To_Maturity'].unique()[0], 6)
    market_ivs = data["mid_iv"]
    expiry = data["Expiry_Date"].iloc[0]

    bren = BrentMethod(max_iter=1000)

    implied_vols = []
    spot_price = data['Coin_Price'].iloc[1]
    moneyness = []

    start_time = datetime.now()
    for idx, row in data.iterrows():
        k = row["Strike_Price"]
        C = row["Market_Price"]
        p_price = row["Put_Price"]
        moneyness_value = np.log(k / spot_price)
        moneyness.append(moneyness_value)

        implied_vol = bren.brent_solution(spot_price=spot_price, strike_price=k, r=risk_free_rate, t=maturities,
                                          market_price=C, moneyness=moneyness_value, put_price=p_price)

        implied_vols.append(implied_vol)

    end_time = datetime.now()
    print("Elasped time", end_time - start_time)

    absolute_errors = [abs(iv - market_iv) for iv, market_iv in zip(implied_vols, market_ivs)]
    mae = np.mean(absolute_errors)
    print(f"Mean Absolute Error (MAE): {mae}")

    plt.plot(moneyness, implied_vols, label='Brent Method Vols')
    plt.plot(moneyness, market_ivs, label='Market Vols')
    plt.xlabel('Log Moneyness')
    plt.ylabel('Implied Volatility')
    plt.title(f"Implied Volatility Smile - {expiry}")
    plt.legend()
    plt.show()

    plt.plot(moneyness, absolute_errors, label="Calibration Absolute Errors")
    plt.xlabel("Log Moneyness")
    plt.ylabel("Absolute Error")
    plt.title(f"Calibration Error (MAE: {mae:.4f})")
    plt.show()
