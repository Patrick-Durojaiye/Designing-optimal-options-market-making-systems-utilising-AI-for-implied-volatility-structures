import numpy as np
from svi_model import SVIModel


class SVISurfaceBuilder:

    def __init__(self, parametised_data):
        self.parametised_data = parametised_data
        self.maturities = self.parametised_data["Maturities"]
        self.market_strikes = self.parametised_data["Strikes"]
        self.spot_price = self.parametised_data['Coin_Price'].iloc[0]
        self.svi_model = SVIModel()

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

        iv_surface = np.zeros((len(self.maturities), len(self.market_strikes)))

        for i, T in enumerate(self.maturities):
            for j, K in enumerate(self.market_strikes):

                x = np.log(K / self.spot_price)

                if T in self.maturities:
                    params = [self.parametised_data["a"], self.parametised_data["b"], self.parametised_data["rho"],
                              self.parametised_data["m"], self.params["sigma"]]

                else:
                    # If T is not in maturities, interpolation is used
                    interpolated_params = []
                    for param_set in zip(*self.params):
                        interpolated_params.append(np.interp(T, self.maturities, param_set))
                    params = interpolated_params

                iv_surface[i, j] = np.sqrt(self.svi_model.evaluate_svi(params, x))

        return iv_surface
