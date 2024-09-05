import pandas as pd
import numpy as np


def get_test_data():

    deribit_chain = pd.read_csv("data/options_market_data_cleaned.csv")
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

    first_timestamp = list(timestamp_dict.keys())[4]
    first_expiry_date = list(timestamp_dict[first_timestamp].keys())[-1]

    data = timestamp_dict[first_timestamp][first_expiry_date]
    return data
