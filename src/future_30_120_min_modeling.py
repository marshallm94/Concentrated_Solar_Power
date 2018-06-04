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


def adj_error_plot(error_dict, colors, title, xlab, ylab, legend_x_loc, legend_y_loc, savefig=False):
    '''
    Plots the errors of two model against each other

    Parameters:
    ----------
    error_dict : (dict)
        A dictionary where the keys are the names of the error arrays
        (i.e 'Linear Regression Error') and the values are an array_like
        (array/list) sequence of errors
    colors : (list)
        List of strings equal to number of keys in error_dict
        (one color for each array of model errors)
    title : (str)
        The title for the plot
    xlab : (str)
        Label for x-axis
    ylab : (str)
        Label for y-axis
    savefig : (bool/str)
        If False default, image will be displayed and not saved. If the
        user would like the image saved, pass the filepath as string to
        which the image should be saved.

    Returns:
    ----------
    None
    '''

    fig, ax = plt.subplots(figsize=(12,8))
    counter = 0

    for name, array in error_dict.items():
        ax.plot(array, '-o', c=colors[counter], label=f"{name}")
        counter +=1

    plt.xticks(range(0,12), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec'])
    plt.yticks(range(15, 301, 20))
    plt.xlabel(xlab, fontweight='bold', fontsize=19)
    plt.ylabel(ylab, fontweight='bold', rotation=0, fontsize=19)
    ax.tick_params(axis='both', labelcolor='black', labelsize=15.0)
    ax.yaxis.set_label_coords(-0.105,0.5)
    plt.suptitle(title, fontweight='bold', fontsize=21)
    plt.legend(bbox_to_anchor=(legend_x_loc, legend_y_loc))
    if savefig:
        plt.savefig(savefig)
    else:
        plt.show()

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

        adj_error_plot(mae_errors, ['orange','blue'], suptitle + " MAE", "Month", r"$\frac{Watts}{Meter^2}$", 1, 1, mae_plot_filename)

        adj_error_plot(rmse_errors, ['orange','blue'], suptitle + " RMSE", "Month", r"$\frac{Watts}{Meter^2}$", 0.29, 0.155, rmse_plot_filename)
