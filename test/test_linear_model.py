#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: 3/4/2022
# =============================================================================

"""
Test suite for the class LM in the package simplerpy
"""

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

        self.empty = LM()

    def test_r_obj_handling(self):
        self.assertRaises(ValueError, self.empty.r_model_obj)

    def test_coefficient_handling(self):
        self.assertRaises(ValueError, self.empty.coefficient)

    def test_df_residual_handling(self):
        self.assertRaises(ValueError, self.empty.df_residual)

    def test_residuals_handling(self):
        self.assertRaises(ValueError, self.empty.residuals)

    def test_standard_error_handling(self):
        self.assertRaises(ValueError, self.empty.standard_error)

    def test_predict_handling(self):
        self.assertRaises(ValueError, self.empty.predict, self.ds2x)

    def test_test_stats_handling(self):
        self.assertRaises(ValueError, self.empty.test_stats)

    def test_p_value_handling(self):
        self.assertRaises(ValueError, self.empty.p_value)

    def test_fitted_values_handling(self):
        self.assertRaises(ValueError, self.empty.fitted_values)

    def test_r_squared_handling(self):
        self.assertRaises(ValueError, self.empty.r_squared)

    def test_adj_r_squared_handling(self):
        self.assertRaises(ValueError, self.empty.adj_r_squared)

    def test_f_statistic_handling(self):
        self.assertRaises(ValueError, self.empty.f_statistic)

    def test_f_test_pvalue_handling(self):
        self.assertRaises(ValueError, self.empty.f_test_pvalue)

    def test_residual_se_handling(self):
        self.assertRaises(ValueError, self.empty.residual_se)

    def test_summary_handling(self):
        self.assertRaises(ValueError, self.empty.summary)

    def test_lm_with_pd(self):
        R_coef = [result[0] for result in R.summary(self.M1R).rx('coefficients')[0]]
        P_coef = self.M1P.coefficient()
        self.assertEqual(R_coef, P_coef)

    def test_lm_with_np_array(self):
        R_coef = [result[0] for result in R.summary(self.M2R).rx('coefficients')[0]]
        P_coef = self.M2P.coefficient()
        self.assertEqual(R_coef, P_coef)

    def test_r_obj(self):
        R_obj = self.M1P.r_model_obj()
        self.assertEqual(type(R_obj), type(self.M1R))

    def test_lm_verbose(self):
        M3P = LM()
        M3P.fit(self.ds2x, self.ds2y, verbose=0)
        R_coef = [result[0] for result in R.summary(self.M2R).rx('coefficients')[0]]
        self.assertEqual(R_coef, M3P.coefficient())

    def test_lm_with_feature_name(self):
        R_coef = [result[0] for result in R.summary(self.M2R).rx('coefficients')[0]]
        M3P = LM()
        M3P.fit(self.ds2x, self.ds2y, feature_name=['x1', 'x2'])
        self.assertEqual(R_coef, M3P.coefficient())

    def test_lm_with_response_name(self):
        R_coef = [result[0] for result in R.summary(self.M2R).rx('coefficients')[0]]
        M3P = LM()
        M3P.fit(self.ds2x, self.ds2y, response_name="test_target")
        self.assertEqual(R_coef, M3P.coefficient())

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
        self.assertEqual(R_resi.tolist(), P_resi)

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
        self.assertEqual(R_fv.tolist(), P_fv)

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
        self.assertEqual(R_fs.tolist(), P_fs)

    def test_f_test_pvalue(self):
        p_value = self.M1P.f_test_pvalue()
        self.assertGreater(p_value, 0.05)

    def test_residual_se(self):
        R_rse = R.summary(self.M1R).rx('sigma')[0][0]
        P_rse = self.M1P.residual_se()
        self.assertEqual(R_rse, P_rse)

    def test_summary(self):
        summary = self.M1P.summary()
        test_summary ='''$coefficients
             Estimate Std. Error   t value  Pr(>|t|)
(Intercept) 0.2727273  1.1634943 0.2344036 0.8364825
x1          0.6909091  0.3534949 1.9545091 0.1898373
x3          1.0909091  1.0204520 1.0690450 0.3969773

Residual standard error: 1.070259 on 2 degrees of freedom
Mutiple R-squared: 0.770909, Adjusted R-squared: 0.541818
F-statistic: 3.365079 on 2.0 and 2.0 DF with p-value: 0.229091'''
        self.assertEqual(test_summary, summary)
