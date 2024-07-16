import numpy as np


class SVIModel:

    def __init__(self):
        pass

    @staticmethod
    def evaluate_svi(params, x):
        a, b, rho, m, sigma = params
        svi_value = a + b * (rho * (x - m) + np.sqrt((x - m)**2 + sigma**2))
        return svi_value
