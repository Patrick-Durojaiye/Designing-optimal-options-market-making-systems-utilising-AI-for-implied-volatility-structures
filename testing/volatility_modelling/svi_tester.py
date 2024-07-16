import numpy as np
from volatility_modelling.calibration_methods.svi_parameters_calibration import SVICalibration
from testing.test_data import get_test_data


def test_svi_calibration():
    data = get_test_data()

    data["Moneyness"] = np.log(data["Strike_Price"] / data["Coin_Price"])

    # Extract unique maturities and strikes
    unique_maturities = data['Time_To_Maturity'].unique()
    unique_strikes = data['Strike_Price'].unique()

    market_vols = np.zeros((len(unique_maturities), len(unique_strikes)))
    data['mid_iv'] = data['mid_iv'] / 100  # converts iv from % to decimal

    # Stores the iv's in market vol
    for i, T in enumerate(unique_maturities):
        for j, K in enumerate(unique_strikes):
            market_vols[i, j] = data[(data['Time_To_Maturity'] == T) & (data['Strike_Price'] == K)]['mid_iv'].values[0]

    strikes = np.array(unique_strikes)
    maturities = np.array(unique_maturities)
    print("maturities", maturities)
    spot_price = data['Coin_Price'].iloc[0]

    svi = SVICalibration(market_strikes=strikes, spot_price=spot_price, market_ivs=market_vols, maturities=maturities)
    svi_params = svi.calibrate()
    print("Svi Params:", svi_params)
    return svi_params

