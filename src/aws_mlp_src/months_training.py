import os
import sys
parentPath = os.path.abspath("..")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)

from train_mlp_base import *
from manipulation import get_master_df

def train_same_months_range(df, num_months):
    results = []
    for day in test_dates:
        X, y = create_X_y(df=df, columns=columns, target='DNI_T_plus15', date=day, num_units=num_months, units='months', same=True)
        train_rmse = train_mlp(X.values, y.values)
        print("\nMLP Training RMSE | {:.4f}\n".format(train_rmse))
        results.append((day, train_rmse))
    return results

def train_months_range(df, num_months):
    results = []
    for day in test_dates:
        X, y = create_X_y(df=df, columns=columns, target='DNI_T_plus15', date=day, num_units=num_months, units='months', same=False)
        train_rmse = train_mlp(X.values, y.values)
        print("\nMLP Training RMSE | {:.4f}\n".format(train_rmse))        
        results.append((day, train_rmse))
    return results

if __name__ == "__main__":
    df = get_master_df("../../data/ivanpah_measurements.csv")
    df['Direct Normal [W/m^2]'] = np.where(df['Direct Normal [W/m^2]'] < 0, 0, df['Direct Normal [W/m^2]'])
    df = create_lagged_DNI_features(15, df)

    one_month_same = train_same_months_range(df, 1)
    two_months_same = train_same_months_range(df, 2)
    three_months_same = train_same_months_range(df, 3)

    one_month_diff = train_months_range(df, 1)
    two_month_diff = train_months_range(df, 2)
    three_month_diff = train_months_range(df, 3)
