from modeling_base import *
from eda import format_nrel_dataframe
from mlp_base import *

def predict_x_mins(shift, row_time_steps=30):
    '''
    Reads and formats NREL data, and creates a target column for modeling
    that is the future values of DNI.

    shift * row_time_steps = number of minutes target column will be in the
    future.

    Example:
        [1]: df = predict_x_mins(shift=4, row_time_steps=30)
            # 4 * 30 = 120. The target will be DNI 120 minutes into
            # the future

    Parameters:
    ----------
    shift : (int)
        The number of observations to shift backward
    row_time_steps : (int)
        The time difference (in minutes) from one observation to the next in
        the DataFrame specified

    Returns:
    ----------
    df : (Pandas DataFrame)
        DataFrame with new column added
    '''
    df = format_nrel_dataframe("../data/2003_2016.csv")
    lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
    df = create_lagged_features(df, lag_features, 4, 30)
    df = create_future_target(df, 'DNI', shift, row_time_steps)
    return df


def separate_mae_rmse(parent_dict):
    '''
    Separates MAE and RMSE error arrays from output of iterative_nn_testing.

    Parameters:
    ----------
    parent_dict : (dict)
        Dictionary that contains error arrays for the Neural Network and
        Persistence model

    Return:
    ----------
    mae_errors : (dict)
        Contains MAE errors for the Neural Network and Persistence Model
    rmse_errors : (dict)
        Contains RMSE errors for the Neural Network and Persistence Model
    '''
    mae_errors = {}
    rmse_errors = {}

    for k, v in parent_dict.items():
        if "MAE" in k:
            mae_errors[k] = v
        elif "RMSE" in k:
            rmse_errors[k] = v

    return mae_errors, rmse_errors


if __name__ == "__main__":

    errors_30_120_min = {}

    for i in range(1, 5):
        target_col = f"DNI_T_plus{i*30}"

        df = predict_x_mins(i)

        max_year = np.unique(df['Year'].values).max()
        min_year = np.unique(df['Year'].values).min()

        test_dates = get_random_test_dates(5, max_year, (4, 20), 2)

        mlp = build_neural_network(len(df.columns) - 3, [10, 40])

        mlp_errors_dict = iterative_nn_testing(mlp, df, target_col, test_dates, max_year - min_year, 'months', fit_params=NN_dict, same=True)

        dates = mlp_errors_dict.pop('date')
        train_obs = mlp_errors_dict.pop('training observations')
        test_obs = mlp_errors_dict.pop('testing observations')

        key = f"{target_col}_errors"
        errors_30_120_min[key] = mlp_errors_dict

    for name, sub_dict in errors_30_120_min.items():
        duration = name.split('_')[2].replace('plus', "")
        suptitle = f"Predicting DNI {duration} minutes Ahead"
        mae_errors, rmse_errors = separate_mae_rmse(sub_dict)

        mae_plot_filename = f"../images/t_plus_{duration}_mae_errors.png"
        rmse_plot_filename = f"../images/t_plus_{duration}_rmse_errors.png"

        error_plot(mae_errors, ['orange','blue'], suptitle + " MAE", "Month", r"$\frac{Watts}{Meter^2}$", 0.3, 0.75, mae_plot_filename)

        error_plot(rmse_errors, ['orange','blue'], suptitle + " RMSE", "Month", r"$\frac{Watts}{Meter^2}$", 0.325, 0.15, rmse_plot_filename)
