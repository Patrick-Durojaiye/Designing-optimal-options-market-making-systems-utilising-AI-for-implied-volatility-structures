import time
import gc
import pandas as pd
import numpy as np
from volatility_modelling.calibration_methods.svi_parameters_calibration import SVICalibration
import multiprocessing
import numpy as np
from functools import partial

def get_surface_data():
    deribit_chain = pd.read_csv("../data/options_market_data_cleaned.csv")
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


def process_expiry(expiry, timestamp_dict):
    data = timestamp_dict[expiry]
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

    return expiry, svi_params


def svi_cal_bench_parallel():
    timestamp_dict, expiry_dates = get_surface_data()


    num_cores = multiprocessing.cpu_count()
    process_expiry_partial = partial(process_expiry, timestamp_dict=timestamp_dict)

    gc.collect()
    start_time = time.perf_counter()

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_expiry_partial, expiry_dates)

    execution_time = time.perf_counter() - start_time

    return execution_time


def run_benchmark(num_trials=100):
    print(f"Number of CPU cores: {multiprocessing.cpu_count()}")

    execution_times = []
    for i in range(num_trials):
        execution_time = svi_cal_bench_parallel()
        execution_times.append(execution_time)

    average_time = np.mean(execution_times)
    std_time = np.std(execution_times)

    print("\nBenchmark Results:")
    print(f"Number of runs: {num_trials}")
    print(f"Average execution time: {average_time:.2f} seconds")
    print(f"Standard deviation: {std_time:.2f} seconds")
    print(f"Average time per expiry: {average_time / 12:.4f} seconds")


if __name__ == '__main__':
    run_benchmark(num_trials=100)





