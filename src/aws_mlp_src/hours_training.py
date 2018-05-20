import os
import sys
import pickle
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from modeling_base import *
from eda import format_nrel_dataframe
from mlp_base import *


def train_hours_range(df, num_hours):
    results = []
    for day in test_dates:
        X, y = create_X_y(df=df, columns=columns, target='DNI_T_plus15', date=day, num_units=num_hours, units='hours', same=False)
        train_rmse = train_mlp(X.values, y.values)
        print("\nMLP Training RMSE | {:.4f}\n".format(train_rmse))
        results.append((day, train_rmse))
    return results


if __name__ == "__main__":

    df = format_nrel_dataframe("../data/2003_2016.csv")
    lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
    df = create_lagged_features(df, lag_features, 4, 30)
    df = create_future_target(df, 'DNI', 1, 30)

    mlp = build_neural_network(len(df.columns) - 3, [10, 40])

    hours_results = {}
    for i in range(10, 70, 10):
        key = f"{i}_hour_same_mlp_results"
        mlp_error_dict = iterative_nn_testing(mlp, df, 'DNI_T_plus30', test_dates, i, 'hours', fit_params=NN_dict, same=True)
        hours_results[key] = mlp_error_dict

    with open('hours_same_results_dict.pickle', 'wb') as f:
        pickle.dump(hours_results, f, protocol=pickle.HIGHEST_PROTOCOL)
