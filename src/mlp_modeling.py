from modeling_base import *
from eda import format_nrel_dataframe
from mlp_base import *

if __name__ == "__main__":

    df = format_nrel_dataframe("../data/2003_2016.csv")
    lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
    df = create_lagged_features(df, lag_features, 4, 30)
    df = create_future_target(df, 'DNI', 1, 30)

    test_dates = get_random_test_dates(5, 2017, (4, 20), 2)

    target_col = 'DNI_T_plus30'
    cols = list(df.columns)
    cols.remove(target_col)

    mlp = build_neural_network(len(cols) - 2, [10, 40])



    X, y = create_X_y(df, cols, 'DNI_T_plus30', test_dates[0], 5, 'months')
    X.drop(['final_date','Date'], axis=1, inplace=True)
    print(X.shape)
    print('DNI_T_plus30' in X.columns)

    mae, rmse, pm_mae, pm_rmse = test_nn_model(mlp, X, y)

    # mlp_error_dict.pop('date')
    #
    # error_plot(rf_error_dict, ['red','orange','blue','green'], 'Neural Network vs Persistence Model Errors', "Cross Validation Period", r"$\frac{Watts}{Meter^2}$", 0.01, 0.375)
