import pandas as pd
import numpy as np
from ast import literal_eval
import matplotlib.pyplot as plt
import matplotlib
from volatility_modelling.models.svi_model import SVIModel

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

def get_svi_data():

    param_data = pd.read_csv("../../data/svi_param_dataset.csv", converters={'moneyness': literal_eval})
    param_data.drop(columns=['Unnamed: 0'], axis=1, inplace=True)

    param_data.set_index('timestamp', inplace=True)
    unique_timestamps = param_data.index.unique()

    timestamp_dict = {}

    for timestamp in unique_timestamps:

        df_timestamp = param_data.loc[timestamp]
        unique_expiry_dates = df_timestamp['Time_To_Maturity'].unique()
        unique_expiry_dates = np.sort(unique_expiry_dates)
        expiry_date_dict = {}

        # Loop through each unique expiry date and filter the data
        for expiry_date in unique_expiry_dates:
            expiry_date_dict[expiry_date] = df_timestamp[df_timestamp['Time_To_Maturity'] == expiry_date]

        # Store the dictionary of expiry dates for the current timestamp
        timestamp_dict[timestamp] = expiry_date_dict

    first_timestamp = list(timestamp_dict.keys())[10]
    expiry_dates = list(timestamp_dict[first_timestamp].keys())

    return timestamp_dict[first_timestamp], expiry_dates


def svi_mae_errors():
    timestamp_dict, expiry_dates = get_surface_data()
    svi_timestamp_dict, svi_expiry_dates = get_svi_data()
    svi = SVIModel()

    mae_per_expiry = []

    all_moneyness = []
    all_residuals = []
    all_expiry_labels = []

    fails = 0
    total_points = 0

    for expiry in expiry_dates:
        print("expiry:", expiry)
        data = timestamp_dict[expiry]
        svi_data = svi_timestamp_dict[expiry]

        market_ivs = data["mid_iv"]
        expiry_date = data["Expiry_Date"].iloc[0]

        moneyness = svi_data["moneyness"].iloc[0]

        a = svi_data['a'].iloc[0]
        b = svi_data['b'].iloc[0]
        rho = svi_data['rho'].iloc[0]
        m = svi_data['m'].iloc[0]
        sigma = svi_data['sigma'].iloc[0]

        implied_vols = np.sqrt([svi.evaluate_svi((a, b, rho, m, sigma), lm) for lm in moneyness])

        # print(implied_vols)

        residuals = [(market_iv - svi_iv) for market_iv, svi_iv in zip(market_ivs, implied_vols)]
        all_moneyness.extend(moneyness)
        all_residuals.extend(residuals)
        all_expiry_labels.extend([expiry] * len(moneyness))

        total_points += len(residuals)
        for error in residuals:
            if error > 0.05 or error < -0.05:
                fails += 1


        absolute_errors = [abs(svi_iv**2 - market_iv**2) for svi_iv, market_iv in zip(implied_vols, market_ivs)]
        mae = np.mean(absolute_errors)
        mae_per_expiry.append(mae)

        rmse = [np.sqrt(((svi_iv**2 - market_iv**2)**2).mean()) for svi_iv, market_iv in zip(implied_vols, market_ivs)]
        print(f"Mean Absolute Error (MAE): {mae}")
        # print(f"RMSE: {rmse}")

        # plt.plot(moneyness, implied_vols, label='SVI Vols')
        # plt.plot(moneyness, market_ivs, label='Market Vols')
        # plt.xlabel('Log Moneyness')
        # plt.ylabel('Implied Volatility')
        # plt.title(f"Implied Volatility Smile - {expiry_date}")
        # plt.legend()
        # plt.show()
        #
        # plt.plot(moneyness, absolute_errors, label="Calibration Absolute Errors")
        # plt.xlabel("Log Moneyness")
        # plt.ylabel("Absolute Error")
        # plt.title(f"Calibration Error (MAE: {mae:.4f})")
        # plt.show()

    print("Avg failure rate", (fails / total_points) * 100)

    plt.scatter(all_moneyness, all_residuals, c=all_expiry_labels, cmap='viridis', alpha=0.7)
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
    plt.colorbar(label='Time To Maturity')
    plt.xlabel("Log Moneyness")
    plt.ylabel("Residuals (Market IV - SVI IV)")
    plt.title("Residuals Across All Maturities")
    plt.legend()
    plt.show()


    print("MAE per expiry", mae_per_expiry)


svi_mae_errors()

