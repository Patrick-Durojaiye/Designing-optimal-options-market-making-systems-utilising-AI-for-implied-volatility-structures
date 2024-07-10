import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib
import sys
matplotlib.use('TkAgg')


class SVIModel:

    def __init__(self, market_strikes: np.array, spot_price: float, market_ivs: np.array, maturities: np.array):
        """
        Initialises the SVI Model with market data

        :param market_strikes: An array of strike prices
        :param spot_price: The current spot price
        :param market_ivs: An array of market implied volatility
        :param maturities: An array of the various expires
        """
        self.market_strikes = market_strikes
        self.spot_price = spot_price
        self.market_ivs = market_ivs
        self.maturities = maturities
        self.log_moneyness = np.log(market_strikes / spot_price)
        self.params = None

    def cost_function(self, params, y, v_tidle):
        c, d, a_tidle = params
        return np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) ** 2)

    def cost_gradient(self, params, y, v_tidle):
        c, d, a_tidle = params

        grad_c = 2 * np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) * np.sqrt(y ** 2 + 1))
        grad_d = 2 * np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle) * y)
        grad_atilde = 2 * np.sum((a_tidle + (d * y) + c * np.sqrt(y ** 2 + 1) - v_tidle))

        return np.array([grad_c, grad_d, grad_atilde])

    def linear_system_solution(self, y, v_tilde):
        A = np.array([
            [np.sum(np.sqrt(y**2 + 1)**2), np.sum(y*np.sqrt(y**2 + 1)), np.sum(np.sqrt(y**2 + 1))],
            [np.sum(y * np.sqrt(y**2 + 1)), np.sum(y**2), np.sum(y)],
            [np.sum(np.sqrt(y**2 +1)), np.sum(y), len(y)]
        ])

        b = np.array([
            [np.sum(v_tilde * np.sqrt(y ** 2 + 1))],
            [np.sum(v_tilde*y)],
            [np.sum(v_tilde)]
        ])

        x = np.linalg.solve(A, b)
        return x

    def check_domain(self, c, d, a_tidle, sigma, v_tilde_max):
        return -sys.float_info.epsilon <= c <= 4*sigma and np.abs(d) <= c and np.abs(d) <= (4*sigma -c) and 0 <= a_tidle <= v_tilde_max

    def gradient_solution(self, params, x, v_tilde):
        m, sigma = params
        y = (x - m) / sigma

        linear_solution = self.linear_system_solution(y=y, v_tilde=v_tilde)
        c = linear_solution[0][0]
        d = linear_solution[1][0]
        a_tilde = linear_solution[2][0]
        v_tilde_max = np.max(v_tilde)
        if self.check_domain(c=c, d=d, a_tidle=a_tilde, sigma=sigma, v_tilde_max=v_tilde_max):
            return self.cost_function(params=[c, d, a_tilde], y=y, v_tidle=v_tilde)
        else:
            initial_guess = np.array([c, d, a_tilde])

            bounds = [(-sys.float_info.epsilon, 4*sigma), (-4*sigma, 4*sigma), (0, v_tilde_max)]
            result = minimize(self.cost_function, initial_guess, args=(y, v_tilde), jac=self.cost_gradient, bounds=bounds, method='L-BFGS-B')
            return result.fun

    def calibrate(self):

        x = self.log_moneyness
        print("x is", x)
        v_tilde = (self.market_ivs**2 * self.maturities[:, np.newaxis]).flatten()
        T = self.maturities.max()

        inital_guess = [0.01, 0.42]
        #m and sigma
        bounds = [(2*min(x), 2*max(x)), (0.005,1)]
        res = minimize(self.gradient_solution, inital_guess, args=(x, v_tilde), bounds=bounds, method="Nelder-Mead")

        m, sigma = res.x
        y = (x - m) / sigma

        linear_solution = self.linear_system_solution(y, v_tilde)
        c = linear_solution[0][0]
        d = linear_solution[1][0]
        a_tilde = linear_solution[2][0]
        print("initial c", c)
        if not self.check_domain(c, d, a_tilde, sigma, np.max(v_tilde)):
            print("not in domain")
            initial_guess = np.array([c, d, a_tilde])

            constraints = [
                {'type': 'ineq', 'fun': lambda x: x[0]},
                {'type': 'ineq', 'fun': lambda x: 4 * sigma - x[0]},
                {'type': 'ineq', 'fun': lambda x: c - abs(x[1])},
                {'type': 'ineq', 'fun': lambda x: 4 * sigma - x[0] - abs(x[1])},
                {'type': 'ineq', 'fun': lambda x: x[2]},
                {'type': 'ineq', 'fun': lambda x: np.max(v_tilde) - x[2]}
            ]

            result = minimize(self.cost_function, initial_guess, args=(y, v_tilde), jac=self.cost_gradient, constraints=constraints, method='L-BFGS-B')
            c, d, a_tilde = result.x

        a = a_tilde / T
        b = c / (sigma*T)
        rho = d / c

        self.params = (a, b, rho, sigma, m)
        return self.params

    def svi(self, params, x):
        """
        Stochastic Volatility Inspired model (SVI)

        :param params: SVI paramaters
        :param x: Log moneyness
        :return:
            float : output of the SVI implied volatility
        """

        a, b, rho, m, sigma = params
        return a + b * (rho * (x - m) + np.sqrt((x - m) ** 2 + sigma ** 2))

    def plot_fit(self):
        """
        Plots the volatility smile from the SVI
        :return:
        """
        if self.params is None:
            raise ValueError("Model parameters have not yet been successfully calibrated.")

        for i, T in enumerate(self.maturities):
            # fitted_vols = np.sqrt(self.svi(self.params[i], self.log_moneyness))
            fitted_vols = np.sqrt([self.svi(self.params, lm)/T for lm in self.market_strikes])
            plt.figure(figsize=(10, 6))
            plt.plot(self.market_strikes, self.market_ivs[i], 'ro', label='Market Vols')
            plt.plot(self.market_strikes, fitted_vols, 'b-', label='SVI Fit')
            plt.xlabel('Strike Prices')
            plt.ylabel('Implied Volatility')
            plt.legend()
            plt.title(f'SVI Calibration Fit to Market Data for Maturity {T} days')
            plt.show()

    def construct_iv_surface(self):
        """
        Constructs an implied volatility surface (IVS)

        This method constructs an IVS from the calibrated SVI parameters. For maturities that are not available the
        method interporlates the parameters.
        :raises:
            ValueError: If the SVI parameters have not yet been successfully calibrated
        :return:
            - iv_surface (np.array) : IVS data stored in an array
        """
        if self.params is None:
            raise ValueError("Model parameters have not yet been successfully calibrated.")

        iv_surface = np.zeros((len(self.maturities), len(self.market_strikes)))

        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.market_strikes):

                x = np.log(K / self.spot_price)

                if T in self.maturities:
                    idx = np.where(self.maturities == T)[0][0]
                    params = self.params[idx]

                else:
                    # If T is not in maturities, interpolation is used
                    interpolated_params = []
                    for param_set in zip(*self.params):
                        interpolated_params.append(np.interp(T, self.maturities, param_set))
                    params = interpolated_params

                iv_surface[i, j] = np.sqrt(self.svi(params, x))

        x, y = np.meshgrid(self.log_moneyness, self.maturities)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, iv_surface, cmap='viridis')
        ax.set_xlabel('Log Moneyness')
        ax.set_ylabel('Maturity')
        ax.set_zlabel('Implied Volatility')
        ax.set_title("Implied Volatility Surface")
        plt.show()
        return iv_surface


import pandas as pd

# data = pd.read_csv("test_data.csv")

deribit_chain = pd.read_csv("options_data_set_4.csv")
deribit_chain.set_index('timestamp', inplace=True)
unique_timestamps = deribit_chain.index.unique()
timestamp_dict = {}

for timestamp in unique_timestamps:

    df_timestamp = deribit_chain.loc[timestamp]

    unique_expiry_dates = df_timestamp['Time_To_Maturity'].unique()
    unique_expiry_dates = np.sort(unique_expiry_dates)

    expiry_date_dict = {}

    # Loops through each unique expiry date and filters the data
    for expiry_date in unique_expiry_dates:
        expiry_date_dict[expiry_date] = df_timestamp[df_timestamp['Time_To_Maturity'] == expiry_date].sort_values(by="Strike_Price", ascending=True)

    # Stores a dictionary of data for each expiry date for the current timestamp
    timestamp_dict[timestamp] = expiry_date_dict


first_timestamp = list(timestamp_dict.keys())[0]
first_expiry_date = list(timestamp_dict[first_timestamp].keys())[0]

data = timestamp_dict[first_timestamp][first_expiry_date]


data["Moneyness"] = np.log(data["Strike_Price"]/data["Coin_Price"])

# Extract unique maturities and strikes
unique_maturities = data['Time_To_Maturity'].unique()
unique_strikes = data['Strike_Price'].unique()

market_vols = np.zeros((len(unique_maturities), len(unique_strikes)))
data['mark_iv'] = data['mark_iv'] / 100  #converts iv from % to decimal

# Stores the iv's in market vol
for i, T in enumerate(unique_maturities):
    for j, K in enumerate(unique_strikes):
        market_vols[i, j] = data[(data['Time_To_Maturity'] == T) & (data['Strike_Price'] == K)]['mark_iv'].values[0]

strikes = np.array(unique_strikes)
maturities = np.array(unique_maturities)

spot_price = data['Coin_Price'].iloc[0]

svi_model = SVIModel(strikes, spot_price, market_vols, maturities)
calibrated_params = svi_model.calibrate()
print("Calibrated SVI Parameters:", calibrated_params)
svi_model.plot_fit()
#svi_model.construct_iv_surface()