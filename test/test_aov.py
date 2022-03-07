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
#         d1 = pd.DataFrame({'x1': [1, 2, 3, 4, 5],
#                                  'x2': [1, 1, 1, 1, 1],
#                                  'x3': [1, 0, 0, 1, 1],
#                                  'x4': [1, 2, 3, 4, 5],
#                                  'y': [2, 1, 3, 5, 4]})

#         self.M1R = R.aov(Formula('y~x1+x2+x3+x4'), d1)
#         self.M1P = AOV()
#         self.M1P.fit('y~x1+x2+x3+x4', d1)
    df = pd.read_csv('Sales_Sample.csv')
    self.M1R = R.aov(\
                     Formula("LAST_SALE_PRICE ~ BEDS + SQFT + LOT_SIZE + BATHS"), df)
    self.M1P = AOV()
    self.M1P.fit('LAST_SALE_PRICE ~ BEDS + SQFT + LOT_SIZE + BATHS', df)

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
    r = sum(list(map(lambda x:pow(x,2), self.M1R.rx2('residuals'))))
    p = self.M1P.sum_of_squares_res()
    self.assertEqual(r, p)

  def test_RSE(self):
    SSResid = sum(list(map(lambda x:pow(x,2), self.M1R.rx2('residuals'))))
    df_resid = self.M1R.rx2('df.residual')[0]
    r = math.sqrt(SSResid/df_resid) 
    p = self.M1P.residual_se()
    self.assertEqual(r, p)

 def test_r_obj(self):
    r = self.M1R
    p = self.M1P.r_model_obj()
    self.assertEqual(type(r), type(p))

  def test_summary(self):
    r = self.M1R
    p = self.M1P.summary()
    rawOut=str(r)
    cleanOut=""
    textOfInterest=False
    for line in rawOut.splitlines():
      if line.startswith("Terms"):
        cleanOut+=line+ "\n"
        textOfInterest=True
      elif textOfInterest:
        cleanOut+=line +"\n"
    self.assertEqual(cleanOut, p)
