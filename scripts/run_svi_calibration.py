import numpy as np
import pandas as pd
from volatility_modelling.calibration_methods.svi_parameters_calibration import SVICalibration

def fetch_data():
    """
    Fetches and processes option data from a CSV file for further analysis.

    This function reads a CSV file containing option data, filters it for call options,
    and organizes the data into a dictionary based on timestamps and expiry dates.

    Returns:
    dict: A nested dictionary where the first level keys are timestamps, and the second level keys are expiry dates.
          Each value is a DataFrame containing the relevant option data for that timestamp and expiry date.
    """

    deribit_chain = pd.read_csv("data/options_market_data_cleaned.csv")

    deribit_chain['Expiry_Date'] = pd.to_datetime(deribit_chain['Expiry_Date'])
    deribit_chain['Expiry_Date'] = deribit_chain['Expiry_Date'].dt.strftime('%d%b%y').str.upper()

    deribit_chain['Strike_Price'] = deribit_chain['Strike_Price'].astype(int).astype(str)

    deribit_chain['instrument_name'] = deribit_chain['Coin_Name'] + '-' + deribit_chain['Expiry_Date'] + '-' + \
                                       deribit_chain['Strike_Price'] + '-C'
    deribit_chain['Strike_Price'] = deribit_chain['Strike_Price'].astype(int)

    # Set 'timestamp' as the index and get unique timestamps
    deribit_chain.set_index('timestamp', inplace=True)
    unique_timestamps = deribit_chain.index.unique()

    # Initialize a dictionary to store the processed data
    timestamp_dict = {}

    for timestamp in unique_timestamps:

        df_timestamp = deribit_chain.loc[timestamp]

        # Get unique expiry dates and sort them
        unique_expiry_dates = df_timestamp['Time_To_Maturity'].unique()
        unique_expiry_dates = np.sort(unique_expiry_dates)
        expiry_date_dict = {}

        # Loop through each unique expiry date and filter the data
        for expiry_date in unique_expiry_dates:
            expiry_date_dict[expiry_date] = df_timestamp[df_timestamp['Time_To_Maturity'] == expiry_date].sort_values(
                by="Strike_Price", ascending=True)

        # Store the dictionary of expiry dates for the current timestamp
        timestamp_dict[timestamp] = expiry_date_dict

    return timestamp_dict


def run_svi_calibration():
    """
    Runs the method of SVI calibration on the fetched option data.

    This function retrieves processed option data, calculates moneyness, and calibrates the SVI model
    for each timestamp and expiry date. The calibration parameters are stored and saved to a CSV file.

    Returns:
    DataFrame: A DataFrame containing the calibration results for each timestamp and expiry date.
    """
    timestamp_dict = fetch_data()

    # Initialize list to store results
    results = []

    for timestamp, expiry_dict in timestamp_dict.items():
        for expiry_date, data in expiry_dict.items():

            data["Moneyness"] = np.log(data["Strike_Price"] / data["Coin_Price"])

            # Extract unique maturities and strikes
            unique_maturities = data['Time_To_Maturity'].unique()
            unique_strikes = data['Strike_Price'].unique()

            market_vols = np.zeros((len(unique_maturities), len(unique_strikes)))

            # Stores the iv's in market vol
            for i, T in enumerate(unique_maturities):
                for j, K in enumerate(unique_strikes):
                    market_vols[i, j] = data[(data['Time_To_Maturity'] == T) & (data['Strike_Price'] == K)]['mid_iv'].values[0]

            strikes = np.array(unique_strikes)
            maturities = np.array(unique_maturities)
            spot_price = data['Coin_Price'].iloc[0]

            # Perform SVI calibration using the fetched data
            svi = SVICalibration(market_strikes=strikes, spot_price=spot_price, market_ivs=market_vols, maturities=maturities)
            svi_params = svi.calibrate()

            # Record the results for the current timestamp and expiry date
            result_record = {
                'timestamp': timestamp,
                'Time_To_Maturity': expiry_date,
                'a': svi_params[0],
                'b': svi_params[1],
                'rho': svi_params[2],
                'm': svi_params[3],
                'sigma': svi_params[4],
                'moneyness': svi.log_moneyness
            }
            results.append(result_record)

    results_df = pd.DataFrame(results)
    results_df["moneyness"] = results_df["moneyness"].apply(lambda x: x.tolist())
    results_df.to_csv("data/svi_param_dataset.csv")

    return results_df
