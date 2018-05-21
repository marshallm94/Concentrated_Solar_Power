from modeling_base import *
from eda import format_nrel_dataframe
from mlp_base import *

if __name__ == "__main__":

    df = format_nrel_dataframe("../data/2003_2016.csv")
    lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
    df = create_lagged_features(df, lag_features, 4, 30)
    df = create_future_target(df, 'DNI', 1, 30)

    test_dates = get_random_test_dates(5, 2016, (4, 20), 2)

    mlp = build_neural_network(len(df.columns) - 3, [10, 40])

    mlp_errors_dict = iterative_nn_testing(mlp, df, 'DNI_T_plus30', test_dates, 5, 'months', fit_params=NN_dict, same=True)

    mlp_errors_dict.pop('date')
    mlp_errors_dict.pop('training observations')
    mlp_errors_dict.pop('testing observations')
    error_plot(mlp_errors_dict, ['red','orange','blue','green'], 'Average Neural Network vs Average Persistence Model Errors', "Month", r"$\frac{Watts}{Meter^2}$", 0.01, 0.375, "../images/nn_v_pm_test_errors.png")    
