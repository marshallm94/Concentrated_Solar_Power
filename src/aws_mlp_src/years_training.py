import os
import sys
import pickle
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from modeling_base import *
from eda import format_nrel_dataframe
from mlp_base import *


if __name__ == "__main__":

    df = format_nrel_dataframe("../../data/2003_2016.csv")
    lag_features = ['Temperature', 'Clearsky DHI', 'Clearsky DNI', 'Clearsky GHI', 'Cloud Type','Dew Point','DHI','DNI','Fill Flag', 'GHI','Relative Humidity','Solar Zenith Angle','Surface Albedo','Pressure','Precipitable Water','Wind Direction','Wind Speed']
    df = create_lagged_features(df, lag_features, 4, 30)
    df = create_future_target(df, 'DNI', 1, 30)

    mlp = build_neural_network(len(df.columns) - 3, [10, 40])

    years_results = {}
    for i in range(1, 8):
        key = f"{i}_year_same_mlp_results"
        mlp_error_dict = iterative_nn_testing(mlp, df, 'DNI_T_plus30', test_dates, i, 'years', fit_params=NN_dict, same=True)
        years_results[key] = mlp_error_dict

    with open('years_results_dict.pickle', 'wb') as f:
        pickle.dump(years_results, f, protocol=pickle.HIGHEST_PROTOCOL)
