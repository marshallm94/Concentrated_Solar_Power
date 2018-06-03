from modeling_base import *
from sklearn.ensemble import RandomForestRegressor
from eda import format_nrel_dataframe
import numpy as np
from mlp_base import *


if __name__ == "__main__":
    df = format_nrel_dataframe("../data/2003_2016.csv")
    lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
    df = create_lagged_features(df, lag_features, 4, 30)
    df = create_future_target(df, 'DNI', 1, 30)

    max_year = np.unique(df['Year'].values).max()
    min_year = np.unique(df['Year'].values).min()

    test_dates = get_random_test_dates(5, max_year, (4, 20), 2)

    rf = RandomForestRegressor()
    mlp = build_neural_network(len(df.columns) - 3, [10, 40])

    rf_error_dict = iterative_testing(rf, df, 'DNI_T_plus30', test_dates, max_year - min_year, 'months', -1, same=True)
    mlp_errors_dict = iterative_nn_testing(mlp, df, 'DNI_T_plus30', test_dates, max_year - min_year, 'months', fit_params=NN_dict, same=True)

    mae_errors = {}
    rmse_errors = {}
    for k, v in rf_error_dict.items():
        if "MAE" in k:
            mae_errors[k] = v
        elif "RMSE" in k:
            rmse_errors[k] = v

    for k, v in mlp_errors_dict.items():
        if "MAE" in k:
            mae_errors[k] = v
        elif "RMSE" in k:
            rmse_errors[k] = v

    error_plot(mae_errors, ['orange','blue','green'], 'Mean Absolute Errors by Model', "Month", r"$\frac{Watts}{Meter^2}$", 0.3, 0.75, "../images/mean_absolute_errors.png")

    error_plot(rmse_errors, ['orange','blue','green'], 'Root Mean Squared Errors by Model', "Month", r"$\frac{Watts}{Meter^2}$", 0.325, 0.15, "../images/root_mean_squared_errors.png")
