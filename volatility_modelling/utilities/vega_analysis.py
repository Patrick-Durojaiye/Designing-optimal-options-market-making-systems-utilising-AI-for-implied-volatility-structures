import matplotlib.pyplot as plt
import matplotlib
from testing.test_data import get_test_data
import numpy as np
matplotlib.use('TkAgg')


def plot_vegas():
    """
    Plots the Vega of options against their log moneyness.

    This function retrieves options data, calculates the log moneyness,
    and plots the Vega of options against the log moneyness. The plot
    displays how the Vega varies with different levels of moneyness for a given expiry date.
    """

    data = get_test_data()
    strike_prices = data["Strike_Price"]
    vegas = data["vega"]
    expiry = data["Expiry_Date"].iloc[0]
    spot_price = data['Coin_Price'].iloc[0]

    moneyness = np.log(strike_prices / spot_price)

    plt.plot(moneyness, vegas)
    plt.xlabel("Log Moneyness log(Strike/Spot)")
    plt.ylabel("Vega")
    plt.title(f"Distribution of Vega for different Log Moneyness, Expiry:{expiry}")
    plt.show()
