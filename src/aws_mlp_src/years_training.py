import os
import sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from train_mlp_base import *
from manipulation import get_master_df

def train_year_range(num_years):
    one_year_results = []
    for day in test_dates:
        X, y = create_X_y(df=df, columns=columns, target='DNI_T_plus15', date=day, num_units=num_years, units='years')
        train_rmse = train_mlp(X.values, y.values)
        one_year_results.append((date, train_rmse))
    return one_year_results


if __name__ == "__main__":
    df = get_master_df("../../data/ivanpah_measurements.csv")
    df = create_lagged_DNI_features(15, df)
    df = df[df['Direct Normal [W/m^2]'] > 0]
    one_year_results = train_year_range(1)
    two_year_results = train_year_range(2)
