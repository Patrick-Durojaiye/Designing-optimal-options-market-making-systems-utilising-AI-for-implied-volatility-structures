from json import loads
from pandas import json_normalize, concat, to_datetime, read_csv
from numpy import where
from requests import get
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import time

url_derebit = 'https://test.deribit.com/api/v2/public/'
url_treasury = 'https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/202407?type=daily_treasury_bill_rates&field_tdr_date_value_month=202407&page&_format=csv'

treasuryBills = (read_csv(url_treasury).iloc[0, 1:] / 100) \
    .transpose() \
    .to_frame(name='Risk_Free_Rate') \
    .assign(Expiry=lambda x: x.index.str.slice(0, 2).astype(int) / 52)

yields = treasuryBills[treasuryBills.index.str.contains('BANK')].reset_index(drop=True)

riskFreeInterp = interp1d(x=yields.Expiry, y=yields.Risk_Free_Rate, fill_value='extrapolate')

def get_time_to_maturity(expiry_date, current_time):
    # Deribit contracts expire at 8am UTC
    expiry_time = expiry_date + timedelta(hours=8)
    time_to_maturity_seconds = (expiry_time - current_time).total_seconds()
    # currently a leap year, every year set to 365
    time_to_maturity_years = time_to_maturity_seconds / (366 * 24 * 60 * 60)
    return time_to_maturity_years

def load_option_prices(coin_name, intervals):
    json_load_names = loads(get(f'{url_derebit}get_instruments?currency={coin_name}&kind=option&expired=false').text)
    results_store = []

    for interval in intervals:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_time = datetime.utcnow()

        for option in set(json_normalize(json_load_names['result'])['instrument_name']):
            json_load_data = loads(get(f'{url_derebit}get_order_book?instrument_name={option}').text)
            option_data = json_normalize(json_load_data['result'])

            option_data['delta'] = option_data['greeks.delta']
            option_data['gamma'] = option_data['greeks.gamma']
            option_data['theta'] = option_data['greeks.theta']
            option_data['vega'] = option_data['greeks.vega']
            option_data['rho'] = option_data['greeks.rho']

            option_data['timestamp'] = timestamp
            results_store.append(option_data)
        time.sleep(interval)
    options_data_set = concat(results_store)[['instrument_name', 'underlying_price', 'mark_price', 'timestamp', 'ask_iv', 'best_ask_price', 'bid_iv', 'best_bid_price', 'delta', 'theta', 'gamma', 'vega', 'rho', 'mark_iv']]
    options_data_set[
        ['Coin_Name', 'Expiry_Date', 'Strike_Price', 'Option_Type']] = \
        options_data_set.instrument_name.str.split('-', expand=True)

    options_data_set = options_data_set.assign(
        Option_Type=where(options_data_set.Option_Type == 'P', 'Put', 'Call'),
        Market_Price=options_data_set.mark_price * options_data_set.underlying_price,
        Strike_Price=options_data_set.Strike_Price.astype(float)
    ).rename({'underlying_price': 'Coin_Price'}, axis='columns').drop(['instrument_name'], axis=1)

    options_data_set['Expiry_Date'] = to_datetime(options_data_set.Expiry_Date, format='%d%b%y')
    options_data_set['Time_To_Maturity'] = options_data_set['Expiry_Date'].apply(lambda expiry_date: get_time_to_maturity(expiry_date, current_time))

    options_data_set = options_data_set.assign(Risk_Free_Rate=riskFreeInterp(options_data_set.Time_To_Maturity))
    options_data_set = options_data_set[options_data_set.Market_Price != 0].reset_index(drop=True)
    return options_data_set


intervals = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
options_data_set = load_option_prices(coin_name="BTC", intervals=intervals)
options_data_set.to_csv('options_data_set_5.csv', index=False)

