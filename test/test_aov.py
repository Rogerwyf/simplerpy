#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michelle Hsieh
# Created Date: 3/6/2022
# =============================================================================

"""
Test suite for the class AOV in the package simplerpy
"""

import math
import unittest

import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects import pandas2ri

from simplerpy.aov import AOV

pandas2ri.activate()
R = ro.r


class AOVTestCase(unittest.TestCase):

    def setUp(self):
        d1 = pd.DataFrame({'x1': [4, 4, 4, 3, 6, 4, 3, 5, 5],
                           'x2': [2.5, 2, 2.25, 2, 2.5, 1.75, 2.75, 3.25,
                                  2.5],
                           'x3': [22578, 4000, 5000, 6400, 7431, 7200,
                                  5500, 12345, 4000],
                           'x4': [2410, 2660, 2800, 3790, 2940, 2240,
                                  3230, 4550, 3800],
                           'y': [678000, 888000, 682000, 1600000, 750000,
                                 682000, 896000, 425000, 911000]})

        self.M1R = R.aov(Formula('y~x1+x2+x3+x4'), d1)
        self.M1P = AOV()
        self.M1P.fit('y~x1+x2+x3+x4', d1)

    def test_df(self):
        r = list((R.summary(self.M1R)[0])['Df'])
        r = r[:-1]
        p = self.M1P.df()
        self.assertEqual(r, p)

    def test_df_residual(self):
        r = self.M1R.rx2('df.residual')[0]
        p = self.M1P.df_residual()
        self.assertEqual(r, p)

    def test_SSR(self):
        r = list((R.summary(self.M1R)[0])['Sum Sq'])
        r = r[:-1]
        p = self.M1P.sum_of_squares()
        self.assertEqual(r, p)

    def test_SSResid(self):
        r = sum(list(map(lambda x: pow(x, 2), self.M1R.rx2('residuals'))))
        p = self.M1P.sum_of_squares_res()
        self.assertEqual(r, p)

    def test_RSE(self):
        SSResid = sum(list(map(lambda x: pow(x, 2), self.M1R.rx2('residuals'))))
        df_resid = self.M1R.rx2('df.residual')[0]
        r = math.sqrt(SSResid / df_resid)
        p = self.M1P.residual_se()
        self.assertEqual(r, p)

    def test_r_obj(self):
        r = self.M1R
        p = self.M1P.r_model_obj()
        self.assertEqual(type(r), type(p))

    def test_summary(self):
        r = "Terms" + "\n" + "              " + "\t" \
            "           x1           x2           x3           x4    Residuals\n" + \
            "Sum of Squares	 2.338845e+11 7.719504e+10 4.396498e+10 2.031836e+11" + \
            " 2.860939e+11\n" \
            + "Deg. of Freedom	            1            1            1            1" + \
            "            4" + "\n\n" + "Residual standard error: 267438.7\n" + \
            "Estimated effects may be unbalanced"
        p = self.M1P.summary()
        self.assertEqual(r, p)
