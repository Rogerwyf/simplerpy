#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: 
# =============================================================================
import unittest

import numpy as np
import pandas as pd
from rpy2 import robjects
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri

from simplerpy.linear_model import LM

pandas2ri.activate()
R = ro.r


class LMTestCase(unittest.TestCase):

    def setUp(self):
        dataset1 = pd.DataFrame({'x1': [1, 2, 3, 4, 5],
                                 'x2': [1, 1, 1, 1, 1],
                                 'x3': [1, 0, 0, 1, 1],
                                 'x4': [1, 2, 3, 4, 5],
                                 'y': [2, 1, 3, 5, 4]})

        dataset2_x = np.array([[2, 1], [1, 2], [2, 2], [2, 3]])
        dataset2_y = np.array([1, 2, 3, 4])

        self.M1R = R.lm('y~x1+x2+x3+x4', data=dataset1)

        robjects.globalenv["y"] = dataset2_y
        robjects.globalenv["x"] = dataset2_x
        self.M2R = R.lm('y~x')

        self.M1P = LM()
        x_train = dataset1.drop(columns=['y'])
        y_train = dataset1['y']
        self.M1P.fit(x_train, y_train, verbose=0)

        self.M2P = LM()
        self.M2P.fit(dataset2_x, dataset2_y,  verbose=0)

        self.ds1 = dataset1
        self.ds2x = dataset2_x
        self.ds2y = dataset2_y

    def test_lm_with_pd(self):
        R_coef = [result[0] for result in R.summary(self.M1R).rx('coefficients')[0]]
        P_coef = self.M1P.coefficient()
        self.assertEqual(R_coef, P_coef)

    def test_lm_with_np_array(self):
        R_coef = [result[0] for result in R.summary(self.M2R).rx('coefficients')[0]]
        P_coef = self.M2P.coefficient()
        self.assertEqual(R_coef, P_coef)

    def test_lm_with_formula(self):
        M1 = R.lm('y~x1+x2+x3', data=self.ds1)
        R_coef = [result[0] for result in R.summary(M1).rx('coefficients')[0]]
        model = LM()
        x_train = self.ds1.drop(columns=['y'])
        y_train = self.ds1['y']
        model.fit(x_train, y_train,formula='y~x1+x2+x3', verbose=0)
        P_coef = model.coefficient()
        self.assertEqual(R_coef, P_coef)

    def test_lm_residual(self):
        R_resi = self.M1R.rx('residuals')[0]
        P_resi = self.M1P.residuals()
        self.assertEqual(R_resi.tolist(), P_resi.tolist())

    def test_lm_df_residual(self):
        R_dfr = self.M1R.rx('df.residual')[0][0]
        P_dfr = self.M1P.df_residual()
        self.assertEqual(R_dfr, P_dfr)

    def test_lm_se(self):
        R_se = [result[1] for result in R.summary(self.M1R).rx('coefficients')[0]]
        P_se = self.M1P.standard_error()
        self.assertEqual(R_se, P_se)

    def test_predict(self):
        X_new = pd.DataFrame({'x1': [2], 'x2': [3], 'x3': [4], 'x4': [5]})
        R_y = R.predict(self.M1R, X_new)
        P_y = self.M1P.predict(X_new)
        self.assertEqual(R_y, P_y)

    def test_test_stats(self):
        R_stats = [result[2] for result in R.summary(self.M1R).rx('coefficients')[0]]
        P_stats = self.M1P.test_stats()
        self.assertEqual(R_stats, P_stats)

    def test_p_value(self):
        R_p = [result[3] for result in R.summary(self.M1R).rx('coefficients')[0]]
        P_p = self.M1P.p_value()
        self.assertEqual(R_p, P_p)

    def test_fitted_values(self):
        R_fv = self.M1R.rx('fitted.values')[0]
        P_fv = self.M1P.fitted_values()
        self.assertEqual(R_fv.tolist(), P_fv.tolist())

    def test_r_squared(self):
        R_rs = R.summary(self.M1R).rx('r.squared')[0][0]
        P_rs = self.M1P.r_squared()
        self.assertEqual(R_rs, P_rs)

    def test_adj_r_squared(self):
        R_rs = R.summary(self.M1R).rx('adj.r.squared')[0][0]
        P_rs = self.M1P.adj_r_squared()
        self.assertEqual(R_rs, P_rs)

    def test_f_statistic(self):
        R_fs = R.summary(self.M1R).rx('fstatistic')[0]
        P_fs = self.M1P.f_statistic()
        self.assertEqual(R_fs.tolist(), P_fs.tolist())

    def test_residual_se(self):
        R_rse = R.summary(self.M1R).rx('sigma')[0][0]
        P_rse = self.M1P.residual_se()
        self.assertEqual(R_rse, P_rse)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(LMTestCase)
    _ = unittest.TextTestRunner().run(suite)


