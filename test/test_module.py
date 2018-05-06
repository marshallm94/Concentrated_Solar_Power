import sys, os
import pandas as pd
import numpy as np
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

        expected_years_set = set([2006, 2007, 2008])
        expected_months_set = set([8])
        test_2008 = pd.date_range("2008-08-01", self.date_string).astype(str).ravel()
        test_2007 = pd.date_range("2007-08-01", "2007-08-31").astype(str).ravel()
        test_2006 = pd.date_range("2006-08-01", "2006-08-31").astype(str).ravel()
        test_dates = np.hstack((np.hstack((test_2006, test_2007)), test_2008))
        test_dates_set = set(test_dates)
        test_dates_set.remove(self.date_string)

        self.assertEqual(set(test_x['Year']), expected_years_set)
        self.assertEqual(set(test_x['Month']), expected_months_set)
        self.assertEqual(set(test_x['Date']), test_dates_set)

    def test_months_2_sameFalse(self):

        test_x, test_y = create_X_y2(df=self.df, columns=self.columns, target=self.target, date=self.date, num_units=self.num_units, units='months', same=False)

        test_dates = pd.date_range("2008-06-13", "2008-08-13").astype(str).ravel()

        test_dates_set = set(test_dates)
        test_dates_set.remove(self.date_string)
        
        self.assertEqual(set(test_x['Date'].values), test_dates_set)



if __name__ == '__main__':
    unittest.main()
