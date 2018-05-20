from modeling import *

df = format_nrel_dataframe("../data/2003_2016.csv")
lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
df = create_lagged_features(df, lag_features, 4, 30)
df = create_future_target(df, 'DNI', 1, 30)

test_dates = tmb.get_random_test_dates(5, 2017, (6, 18), 2)

rf = RandomForestRegressor()
rf_error_dict = iterative_testing(rf, df, 'DNI_T_plus30', test_dates, 5, 'months', same=True)
rf_error_dict.pop('date')
error_plot(rf_error_dict, ['red','orange','blue','green'], 'Random Forest vs Persistence Model Errors', "Cross Validation Period", r"$\frac{Watts}{Meter^2}$", 0.01, 0.375, "../images/rf_v_pm_cv_errors.png")