import numpy as np
from volatility_modelling.models.svi_model import SVIModel
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

class SVISurfaceBuilder:
    """
    A class for plotting the implied volatility surface from the Stochastic Volatility Inspired (SVI).
    """
    def __init__(self, parametised_data):
        """
        Initialize the SVISurfaceBuilder object with market data.

        :param parametised_data: Data of SVI parameters
        """
        self.parametised_data = parametised_data
        self.maturities = self.parametised_data.index.tolist()
        self.moneyness = self.parametised_data["moneyness"]
        self.svi_model = SVIModel()

    def construct_iv_surface(self):
        """
        Constructs an implied volatility surface (IVS)

        This method constructs an IVS from the calibrated SVI parameters using the original moneyness values.

        :return:
            - iv_surface (list of np.arrays) : IVS data stored as a list of arrays, one for each maturity
            - moneyness_list (list of np.arrays) : Original moneyness values for each maturity
        """

        iv_surface = []
        moneyness_list = []

        for i, T in enumerate(self.maturities):
            params = [self.parametised_data["a"].iloc[i], self.parametised_data["b"].iloc[i],
                      self.parametised_data["rho"].iloc[i],
                      self.parametised_data["m"].iloc[i], self.parametised_data["sigma"].iloc[i]]

            current_moneyness = self.moneyness.iloc[i]
            current_iv = []

            for lm in current_moneyness:
                try:
                    svi_value = self.svi_model.evaluate_svi(params, lm)
                    iv = np.sqrt(svi_value)
                    current_iv.append(iv)
                except Exception as e:
                    print(f"Error at T={T}, lm={lm}, params={params}: {str(e)}")
                    current_iv.append(np.nan)

            iv_surface.append(np.array(current_iv))
            moneyness_list.append(np.array(current_moneyness))

        return iv_surface, moneyness_list

    def plot_iv_surface(self):
        """
        Plots the implied volatility surface
        """
        iv_surface, moneyness_list = self.construct_iv_surface()

        X_flat = np.concatenate([np.repeat(m, len(moneyness)) for m, moneyness in zip(self.maturities, moneyness_list)])
        Y_flat = np.concatenate(moneyness_list)
        Z_flat = np.concatenate(iv_surface)

        X_grid, Y_grid = np.meshgrid(np.linspace(min(Y_flat), max(Y_flat), 100),
                                     np.linspace(min(X_flat), max(X_flat), 100))

        Z_grid = griddata((Y_flat, X_flat), Z_flat, (X_grid, Y_grid), method="cubic")

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap="coolwarm")

        ax.set_xlabel("Log Moneyness log(K/S)")
        ax.set_ylabel("Time to Maturity")
        ax.set_zlabel("IV")
        ax.set_title("BTC Implied Volatility Surface")
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()
