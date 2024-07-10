import numpy as np
from PricingModel import BlackScholes
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

class NewthonRaphson:

    def __init__(self, max_iter: int):
        self.max_iter = max_iter
        self.bs = BlackScholes()

    def nr_solution(self, spot_price: float, strike_price: float, r: float, initial_guess: float, t: float, market_price: float, tolerance=1e-8):

        epsilon = 1e-4
        sigma = initial_guess

        for i in range(self.max_iter+1):

            bs_price = self.bs.call_value(spot_price=spot_price, strike_price=strike_price, r=r, sigma=sigma, t=t)
            bs_price = round(bs_price, 8)
            f = bs_price - market_price
            f_prime = self.bs.vega(spot_price=spot_price, t=t, strike_price=strike_price, r=r, sigma=sigma)

            if abs(f_prime) < epsilon:
                break

            new_guess = sigma - (f / f_prime)

            if abs(new_guess - sigma) < tolerance:
                return new_guess

            sigma = new_guess

        return np.nan


# Example of Newton Raphson Calibration
deribit_chain = pd.read_csv("options_data_set_4.csv")
deribit_chain = deribit_chain[deribit_chain["Option_Type"] == "Call"]
deribit_chain['Expiry_Date'] = pd.to_datetime(deribit_chain['Expiry_Date'])
deribit_chain['Expiry_Date'] = deribit_chain['Expiry_Date'].dt.strftime('%d%b%y').str.upper()
deribit_chain['Strike_Price'] = deribit_chain['Strike_Price'].astype(int).astype(str)
deribit_chain['instrument_name'] = deribit_chain['Coin_Name'] + '-' + deribit_chain['Expiry_Date'] + '-' + deribit_chain['Strike_Price'] + '-C'
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
        expiry_date_dict[expiry_date] = df_timestamp[df_timestamp['Time_To_Maturity'] == expiry_date].sort_values(by="Strike_Price", ascending=True)

    # Store the dictionary of expiry dates for the current timestamp
    timestamp_dict[timestamp] = expiry_date_dict

first_timestamp = list(timestamp_dict.keys())[0]
first_expiry_date = list(timestamp_dict[first_timestamp].keys())[0]

data = timestamp_dict[first_timestamp][first_expiry_date]

risk_free_rate = round(data['Risk_Free_Rate'].iloc[0],6)
strike_prices = data["Strike_Price"]
maturities = round(data['Time_To_Maturity'].unique()[0], 6)
market_ivs = data["mark_iv"] / 100

nr = NewthonRaphson(max_iter=100000)

implied_vols = []
initial_guess = 1.03

for idx, row in data.iterrows():
    k = row["Strike_Price"]
    C = row["Market_Price"]
    spot_price = row['Coin_Price']
    implied_vol = nr.nr_solution(spot_price=spot_price, strike_price=k, r=risk_free_rate, initial_guess=initial_guess, t=maturities, market_price=C)
    implied_vols.append(implied_vol)



print(implied_vols)
plt.plot(strike_prices, implied_vols, label='Newton Raphson Vols')
plt.plot(strike_prices, market_ivs,  label='Market Vols')
plt.xlabel('Strike Prices')
plt.ylabel('Implied Volatility')
plt.legend()
plt.show()


