import pandas as pd
import numpy as np

from simplerpy.t_test import tTest
from rpy2.robjects.packages import importr
base = importr('base')
stats = importr('stats')

import unittest

class tTestCase(unittest.TestCase):
    def setUp(self):
        dataset1 = pd.DataFrame({
            'x': [3,2,5,0,9,10],
            'y': [12,3,6,10,2,4]
        })

        dataA = np.array([2,3,4,5,6])
        dataB = np.array([0,9,3,2,4])

        d_1 = dataset1['x']
        d_2 = dataset1['y']

        #one sample test with default values
        # pandas
        self.M1R_one = stats.t_test(d_1)
        self.M1P_one = tTest()
        self.M1P_one.fit(d_1)
        # numpy
        self.M1R_one_np = stats.t_test(dataA)
        self.M1P_one_np = tTest()
        self.M1P_one_np.fit(dataA)

        #two sample test with default values
        # pandas
        self.M2R_two = stats.t_test(d_1, d_2)
        self.M2P_two = tTest()
        self.M2P_two.fit(d_1, d_2)
        # numpy
        self.M2R_two_np = stats.t_test(dataA, dataB)
        self.M2P_two_np = tTest()
        self.M2P_two_np.fit(dataA, dataB)

    def test_t_with_pd_one(self):
        R_testval = self.M1R_one
        P_testval = self.M1P_one.r_model_obj()
        self.assertEqual(type(R_testval), type(P_testval))

    def test_t_with_np_one(self):
        R_testval = self.M1R_one_np
        P_testval = self.M1P_one_np.r_model_obj()
        self.assertEqual(type(R_testval), type(P_testval))

    def test_t_with_pd_two(self):
        R_testval_two = self.M2R_two
        P_testval_two = self.M2P_two.r_model_obj()
        self.assertEqual(type(R_testval_two), type(P_testval_two))

    def test_t_with_np_two(self):
        R_testval_np = self.M2R_two_np
        P_testval_np = self.M2P_two_np.r_model_obj()
        self.assertEqual(type(R_testval_np), type(P_testval_np))

    def test_t_val(self):
        R_testval_two = self.M2R_two.rx2('statistic')[0]
        P_testval_two = self.M2P_two.tvalue()
        self.assertEqual(R_testval_two, P_testval_two)

    def test_t_pvalue(self):
        R_pval = self.M2R_two.rx2('p.value')[0]
        P_pval = self.M2P_two.pvalue()
        self.assertEqual(R_pval, P_pval)

    def test_t_df(self):
        R_df = self.M2R_two.rx2('parameter')[0]
        P_df = self.M2P_two.df()
        self.assertEqual(R_df, P_df)

    def test_t_ci(self):
        R_ci = self.M2R_two.rx2('conf.int')[:2]
        P_ci = self.M2P_two.ci()
        self.assertEqual(R_ci.any(), P_ci.any())

    def test_t_estimate(self):
        R_est = self.M2R_two.rx2('estimate')
        P_est = self.M2P_two.estimate()
        self.assertEqual(R_est.any(), P_est.any())

    def test_t_stderror(self):
        R_stder = self.M2R_two.rx2('stderr')[0]
        P_stder = self.M2P_two.stderror()
        self.assertEqual(R_stder, P_stder)

    def test_t_alternative(self):
        R_alt = self.M2R_two.rx2('alternative')[0]
        P_alt = self.M2P_two.alternative()
        self.assertEqual(R_alt, P_alt)

    def test_t_method(self):
        R_method = self.M2R_two.rx2('method')[0]
        P_method = self.M2P_two.method()
        self.assertEqual(R_method, P_method)

    def test_t_summary(self):
        temp = str(self.M1R_one)
        index_of_d = temp.index('data')
        index_of_t = temp[index_of_d + 15:].index('t')
        R_sum = temp[:index_of_d] + temp[index_of_d + 15 + index_of_t:]
        P_sum = self.M1P_one.summary()
        self.assertEqual(R_sum, P_sum)

