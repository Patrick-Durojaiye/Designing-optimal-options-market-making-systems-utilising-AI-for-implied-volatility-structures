import schedule
import time
import logging
import os
from json import loads
from pandas import json_normalize, concat, to_datetime, read_csv
from numpy import where
from requests import get, RequestException
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set up logging
logging.basicConfig(filename='option_data_collection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

url_derebit = 'https://deribit.com/api/v2/public/'
url_treasury = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/202407?type=daily_treasury_bill_rates&field_tdr_date_value_month=202407&page&_format=csv'

treasuryBills = (read_csv(url_treasury).iloc[0, 1:] / 100) \
    .transpose() \
    .to_frame(name='Risk_Free_Rate') \
    .assign(Expiry=lambda x: x.index.str.slice(0, 2).astype(int) / 52)

yields = treasuryBills[treasuryBills.index.str.contains('BANK')].reset_index(drop=True)
riskFreeInterp = interp1d(x=yields.Expiry, y=yields.Risk_Free_Rate, fill_value='extrapolate')


def get_time_to_maturity(expiry_date):
    days_to_maturity = (expiry_date.date() - datetime.now().date()).days +1
    days_to_maturity = days_to_maturity / 365
    return days_to_maturity


def fetch_option_data(option, timestamp, max_retries=3):
    for attempt in range(max_retries):
        try:
            json_load_data = loads(get(f'{url_derebit}get_order_book?instrument_name={option}').text)
            if 'result' not in json_load_data:
                raise ValueError(f"Unexpected API response for {option}")

            option_data = json_normalize(json_load_data['result'])
            option_data['delta'] = option_data['greeks.delta']
            option_data['gamma'] = option_data['greeks.gamma']
            option_data['theta'] = option_data['greeks.theta']
            option_data['vega'] = option_data['greeks.vega']
            option_data['rho'] = option_data['greeks.rho']
            option_data['timestamp'] = timestamp
            return option_data
        except (RequestException, ValueError) as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to fetch data for {option} after {max_retries} attempts: {str(e)}")
                return None
            time.sleep(1)  # Wait before retrying


def load_option_prices(coin_name):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    json_load_names = loads(get(f'{url_derebit}get_instruments?currency={coin_name}&kind=option&expired=false').text)
    options = set(json_normalize(json_load_names['result'])['instrument_name'])

    results_store = []
    failed_options = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_option = {executor.submit(fetch_option_data, option, timestamp): option for option in options}
        for future in as_completed(future_to_option):
            option = future_to_option[future]
            try:
                option_data = future.result()
                if option_data is not None:
                    results_store.append(option_data)
                else:
                    failed_options.append(option)
            except Exception as exc:
                logging.error(f"{option} generated an exception: {exc}")
                failed_options.append(option)

    if failed_options:
        logging.warning(f"Failed to fetch data for {len(failed_options)} options: {failed_options}")

    if not results_store:
        logging.error("No data collected in this interval")
        return None

    options_data_set = concat(results_store)[
        ['instrument_name', 'underlying_price', 'mark_price', 'timestamp', 'ask_iv', 'best_ask_price', 'bid_iv',
         'best_bid_price', 'delta', 'theta', 'gamma', 'vega', 'rho', 'mark_iv']]

    options_data_set[['Coin_Name', 'Expiry_Date', 'Strike_Price', 'Option_Type']] = \
        options_data_set.instrument_name.str.split('-', expand=True)

    options_data_set = options_data_set.assign(
        Option_Type=where(options_data_set.Option_Type == 'P', 'Put', 'Call'),
        Market_Price=options_data_set.mark_price * options_data_set.underlying_price,
        Strike_Price=options_data_set.Strike_Price.astype(float)
    ).rename({'underlying_price': 'Coin_Price'}, axis='columns').drop(['instrument_name'], axis=1)

    options_data_set['Expiry_Date'] = to_datetime(options_data_set.Expiry_Date, format='%d%b%y')
    options_data_set['Time_To_Maturity'] = options_data_set['Expiry_Date'].apply(
        lambda expiry_date: get_time_to_maturity(expiry_date))

    options_data_set = options_data_set.assign(Risk_Free_Rate=riskFreeInterp(options_data_set.Time_To_Maturity))

    return options_data_set


def append_to_csv(df, filename='options_dataset_combined.csv'):
    """
    Append dataframe to CSV file, create if it doesn't exist.
    """
    if not os.path.isfile(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, mode='a', header=False, index=False)


def job():
    try:
        options_data_set = load_option_prices(coin_name="BTC")
        if options_data_set is not None and not options_data_set.empty:
            # Add a timestamp column to the dataframe
            options_data_set['collection_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Append to the combined CSV file
            append_to_csv(options_data_set)

            logging.info(f"Data collected and appended to options_dataset_combined.csv")
        else:
            logging.warning("No data collected in this interval")
    except Exception as e:
        logging.error(f"An error occurred during data collection: {str(e)}")


# Schedule the job to run every 5 minutes
schedule.every(2).minutes.until(datetime.now() + timedelta(hours=11)).do(job)
max_iterations = 320

# Run the scheduled jobs
if __name__ == "__main__":

    while True:
        schedule.run_pending()
