import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon
from volatility_modelling.models.svi_model import SVIModel
from volatility_modelling.utilities.svi_calibration_error_test import get_surface_data
from volatility_modelling.utilities.svi_calibration_error_test import get_svi_data
from volatility_modelling.calibration_methods.bent_calibration import BrentMethod



def hypotehsis_test():

    timestamp_dict, expiry_dates = get_surface_data()
    svi_timestamp_dict, svi_expiry_dates = get_svi_data()

    svi = SVIModel()
    bren = BrentMethod(max_iter=1000)

    svi_mae_per_expiry = []
    brent_mae_per_expiry = []

    for expiry in expiry_dates:
        print("expiry:", expiry)
        data = timestamp_dict[expiry]
        svi_data = svi_timestamp_dict[expiry]


        risk_free_rate = round(data['Risk_Free_Rate'].iloc[0], 6)
        maturities = round(data['Time_To_Maturity'].unique()[0], 6)
        market_ivs = data["mid_iv"]
        expiry_date = data["Expiry_Date"].iloc[0]



        brent_implied_vols = []
        spot_price = data['Coin_Price'].iloc[1]

        svi_moneyness = svi_data["moneyness"].iloc[0]

        a = svi_data['a'].iloc[0]
        b = svi_data['b'].iloc[0]
        rho = svi_data['rho'].iloc[0]
        m = svi_data['m'].iloc[0]
        sigma = svi_data['sigma'].iloc[0]

        svi_implied_vols = np.sqrt([svi.evaluate_svi((a, b, rho, m, sigma), lm) for lm in svi_moneyness])

        for idx, row in data.iterrows():
            k = row["Strike_Price"]
            C = row["Market_Price"]
            put_price = row["Put_Price"]
            moneyness_value = np.log(k / spot_price)
            brent_implied_vol = bren.brent_solution(spot_price=spot_price, strike_price=k, r=risk_free_rate, t=maturities,
                                              market_price=C, moneyness=moneyness_value, put_price=put_price)
            brent_implied_vols.append(brent_implied_vol)


        svi_absolute_errors = [abs(svi_iv ** 2 - market_iv ** 2) for svi_iv, market_iv in zip(svi_implied_vols, market_ivs)]
        svi_mae = np.mean(svi_absolute_errors)
        svi_mae_per_expiry.append(svi_mae)

        brent_absolute_errors = [abs(brent_iv ** 2 - market_iv ** 2) for brent_iv, market_iv in zip(brent_implied_vols, market_ivs)]
        brent_mae = np.mean(brent_absolute_errors)
        brent_mae_per_expiry.append(brent_mae)

    svi_mae_per_expiry = np.array(svi_mae_per_expiry)
    brent_mae_per_expiry = np.array(brent_mae_per_expiry)

    mae_difference = svi_mae_per_expiry - brent_mae_per_expiry

    shapiro_statistic, shapiro_p_value = shapiro(mae_difference)
    print("Shapiro-Wilk test statistic", shapiro_statistic)
    print("Shapiro-Wilk test p-value", shapiro_p_value)

    # Null hypothesis is that the mean of the absolute differences is 0, No difference between SVI and Brent
    if shapiro_p_value > 0.05:

        t_statistic, t_test_p_value = ttest_rel(svi_mae_per_expiry, brent_mae_per_expiry, alternative="less")
        print("Paired T-test Statistic", t_statistic)
        print("Paired T-test P Value", t_test_p_value)

        if t_test_p_value < 0.05:
            print("Null hypothesis rejected, SVI performs better than Brent")
        else:
            print("Failed to reject the null hypothesis, no significant difference between SVI and Brent")

    else:
        w_statistic, w_p_value = wilcoxon(svi_mae_per_expiry, brent_mae_per_expiry, alternative="less")
        print("Wilcoxon test statistic: ", w_statistic)
        print("Wilcoxon test p-value: ", w_p_value)

        if w_p_value < 0.05:
            print("Null hypothesis rejected, SVI performs better than Brent")
        else:
            print("Failed to reject the null hypothesis, no significant difference between SVI and Brent")

hypotehsis_test()
