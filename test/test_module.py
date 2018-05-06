import sys, os
import pandas as pd
import unittest

testdir = os.path.dirname(__file__)
srcdir = '../src/'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))

from modeling import create_X_y2, create_lagged_DNI_features
from manipulation import get_master_df

class TestCreateXY(unittest.TestCase):

    def setUp(self):

        self.date_string = "2008-08-13"
        self.date = pd.to_datetime(self.date_string)
        self.start_df = get_master_df("../data/ivanpah_measurements.csv")
        self.df = create_lagged_DNI_features(15, self.start_df)
        self.columns = ['Year','DOY','Month','Hour','Minute','Date','final_date','DNI_T_minus9', 'DNI_T_minus6']
        self.target = 'DNI_T_minus15'
        self.num_units = 2


    def test_months_2_sameTrue(self):

        test_x, test_y = create_X_y2(df=self.df, columns=self.columns, target=self.target, date=self.date, num_units=self.num_units, units='months', same=True)

        mask = self.df['Year'] >= self.date.year - self.num_units
        mask2 = self.df['Year'] <= self.date.year
        mask3 = self.df['Month'] == self.date.month
        mask4 = self.df['final_date'] < self.date
        expected_y = self.df[mask & mask2 & mask3 & mask4][self.target].values
        expected_x = self.df[mask & mask2 & mask3 & mask4][self.columns].values
        print(expected_x.shape, expected_y.shape)
        print(test_x.shape, test_y.shape)
        self.assertEqual(test_x.shape, expected_x.shape)

    def test_months_2_sameFalse(self):

        test_x, test_y = create_X_y2(df=self.df, columns=self.columns, target=self.target, date=self.date, num_units=self.num_units, units='months', same=False)

                


if __name__ == '__main__':
    unittest.main()
