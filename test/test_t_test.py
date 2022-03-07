import pandas as pd

from simplerpy.t_test import tTest
from rpy2.robjects.package import importr
base = importr('base')
stats = importr('stats')

import unittest

class tTestCase(unittest.TestCase):
    def setUp(self):
        dataset1 = pd.DataFrame({
            'x': [3,2,5,0,9,10]
            'y': [12,3,6,10,2,4]
        })

        dataset2 = np.array([2,3,4,5,6])
        dataset3 = np.array([0,9,3,2,4])

        d_1 = dataset1['x']
        d_2 = dataset1['y']
        self.M1R = stats.t_test(d_1, d_2)
        self.M2R = tTest()
        self.M2R.fit(d_1, d_2)
        pass

    def test_t_with_pd(self):
        pass

    def test_t_with_np_array(self):
        pass

    def test_t_pvalue(self):
        pass

    def test_t_tvalue(self):
        pass

    def test_t_df(self):
        pass

    def test_t_ci(self):
        pass

    def test_t_estimate(self):
        pass

    def test_t_stderror(self):
        pass

    def test_t_alternative(self):
        pass

    def test_t_method(self):
        pass

    def test_t_summary(self):
        pass