from scripts.run_svi_calibration import run_svi_calibration
from testing.volatility_modelling.bent_tester import test_brent_method
from testing.volatility_modelling.newton_raphson_tester import test_newton_raphson
from testing.volatility_modelling.svi_tester import test_svi_calibration
from volatility_modelling.utilities.vega_analysis import plot_vegas


def main():
    # plot_vegas()
    # test_svi_calibration()
    # test_newton_raphson(1.0)
    # test_brent_method()
    run_svi_calibration()

if __name__ == '__main__':
    main()
