from modeling_base import *
from sklearn.ensemble import RandomForestRegressor
from eda import format_nrel_dataframe
import numpy as np

if __name__ == "__main__":
    df = format_nrel_dataframe("../data/2003_2016.csv")
    lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
    df = create_lagged_features(df, lag_features, 4, 30)
    df = create_future_target(df, 'DNI', 1, 30)

    max_year = np.unique(df['Year'].values).max()
    min_year = np.unique(df['Year'].values).min()

    test_dates = get_random_test_dates(5, max_year, (4, 20), 2)

    rf = RandomForestRegressor()
    rf_error_dict = iterative_testing(rf, df, 'DNI_T_plus30', test_dates, max_year - min_year, 'months', -1, same=True)

    dates = rf_error_dict.pop('date')

    error_plot(rf_error_dict, ['red','orange','blue','green'], 'Average Random Forest vs Average Persistence Model Errors', "Month", r"$\frac{Watts}{Meter^2}$", 0.053, 0.375, "../images/rf_v_pm_test_errors.png")    
