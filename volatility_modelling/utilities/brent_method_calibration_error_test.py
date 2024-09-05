import pandas as pd
import numpy as np

from volatility_modelling.calibration_methods.bent_calibration import BrentMethod
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def get_surface_data():
    deribit_chain = pd.read_csv("../../data/options_market_data_cleaned.csv")
    deribit_chain['Expiry_Date'] = pd.to_datetime(deribit_chain['Expiry_Date'])
    deribit_chain['Expiry_Date'] = deribit_chain['Expiry_Date'].dt.strftime('%d%b%y').str.upper()
    deribit_chain['Strike_Price'] = deribit_chain['Strike_Price'].astype(int).astype(str)
    deribit_chain['instrument_name'] = deribit_chain['Coin_Name'] + '-' + deribit_chain['Expiry_Date'] + '-' + \
                                       deribit_chain['Strike_Price'] + '-C'
    deribit_chain['Strike_Price'] = deribit_chain['Strike_Price'].astype(int)

    deribit_chain.set_index('timestamp', inplace=True)
    unique_timestamps = deribit_chain.index.unique()
    timestamp_dict = {}

    for timestamp in unique_timestamps:

        df_timestamp = deribit_chain.loc[timestamp]

        unique_expiry_dates = df_timestamp['Time_To_Maturity'].unique()
        unique_expiry_dates = np.sort(unique_expiry_dates)
        expiry_date_dict = {}

        # Loop through each unique expiry date and filter the data
        for expiry_date in unique_expiry_dates:
            expiry_date_dict[expiry_date] = df_timestamp[df_timestamp['Time_To_Maturity'] == expiry_date].sort_values(
                by="Strike_Price", ascending=True)

        # Store the dictionary of expiry dates for the current timestamp
        timestamp_dict[timestamp] = expiry_date_dict

    first_timestamp = list(timestamp_dict.keys())[10]
    expiry_dates = list(timestamp_dict[first_timestamp].keys())

    return timestamp_dict[first_timestamp], expiry_dates


def brent_mae_errors():
    timestamp_dict, expiry_dates = get_surface_data()

    mae_per_expiry = []

    all_moneyness = []
    all_residuals = []
    all_expiry_labels = []

    fails = 0
    total_points = 0

    for expiry in expiry_dates:
        print("expiry:", expiry)
        data = timestamp_dict[expiry]

        risk_free_rate = round(data['Risk_Free_Rate'].iloc[0], 6)
        maturities = round(data['Time_To_Maturity'].unique()[0], 6)
        market_ivs = data["mid_iv"]
        expiry_date = data["Expiry_Date"].iloc[0]

        bren = BrentMethod(max_iter=1000)

        implied_vols = []
        spot_price = data['Coin_Price'].iloc[1]
        moneyness = []

        for idx, row in data.iterrows():
            k = row["Strike_Price"]
            C = row["Market_Price"]
            put_price = row["Put_Price"]
            moneyness_value = np.log(k / spot_price)
            moneyness.append(moneyness_value)
            implied_vol = bren.brent_solution(spot_price=spot_price, strike_price=k, r=risk_free_rate, t=maturities,
                                              market_price=C, moneyness=moneyness_value, put_price=put_price)

            implied_vols.append(implied_vol)

        residuals = [(market_iv - brent_iv) for market_iv, brent_iv in zip(market_ivs, implied_vols)]
        all_moneyness.extend(moneyness)
        all_residuals.extend(residuals)
        all_expiry_labels.extend([expiry] * len(moneyness))

        absolute_errors = [abs(iv - market_iv) for iv, market_iv in zip(implied_vols, market_ivs)]

        total_points += len(residuals)
        for error in residuals:
            if error > 0.05 or error < -0.05:
                fails += 1

        mae = np.mean(absolute_errors)
        mae_per_expiry.append(mae)
        print(f"Mean Absolute Error (MAE): {mae}")

        plt.plot(moneyness, implied_vols, label='Brent Method Vols')
        plt.plot(moneyness, market_ivs, label='Market Vols')
        plt.xlabel('Log Moneyness')
        plt.ylabel('Implied Volatility')
        plt.title(f"Implied Volatility Smile - {expiry_date}")
        plt.legend()
        plt.show()

        plt.plot(moneyness, absolute_errors, label="Calibration Absolute Errors")
        plt.xlabel("Log Moneyness")
        plt.ylabel("Absolute Error")
        plt.title(f"Calibration Error (MAE: {mae:.4f})")
        plt.show()

    print("Avg failure rate", (fails/total_points)*100)

    plt.scatter(all_moneyness, all_residuals, c=all_expiry_labels, cmap='viridis', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
    plt.colorbar(label='Time To Maturity')
    plt.xlabel("Log Moneyness")
    plt.ylabel("Residuals (Market IV - Brent Method IV)")
    plt.title("Residuals Across All Maturities")
    plt.legend()
    plt.show()

    print("MAE per expiry", mae_per_expiry)


brent_mae_errors()
