#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Michelle Hsieh
# Created Date: 3/5/2022
# =============================================================================

"""
The class 'AOV' contains an ordinary linear model object (aov) from R
through the bridge package rpy2. This class mimics how most machine learning
models in Python work and provide extra information retrieved from the R
object.
"""

import math

from rpy2 import robjects as ro
from rpy2.robjects import Formula
from rpy2.robjects import pandas2ri

pandas2ri.activate()
R = ro.r


class AOV:
    def __init__(self):
        """
        Initialize the AOV object, fit() must be called before calling all
        other methods of this object.

        """
        self._model = None

    def fit(self, f, df):
        """
        Fits the aov model with input formula using aov() in R
        through the package rpy2.
        :param f: formula used for aov() i.e. "y~x"
        :param df: the data frame containing variables of interest
        :return: None, assign the R model object to self._model
        """
        self._model = R.aov(Formula(f), df)

    def r_model_obj(self):
        """
        Returns the fitted R model object.

        :return: a R model object
        """
        return self._model

    def df(self):
        """
        Returns the degrees of freedom on the feature variables of the
        fitted model, retrieved from the R model object.

        :return: list of integers
        """
        sumOut = R.summary(self._model)[0]
        degFree = list(sumOut['Df'])
        return degFree[:-1]

    def df_residual(self):
        """
        Returns the degrees of freedom on residuals of the fitted model
        , retrieved from the R model object.

        :return: an integer
        """
        df = self._model.rx2('df.residual')[0]
        return df

    def sum_of_squares(self):
        """

        Returns the sum of squares based on the fitted model, retrieved
        from the R model object.

        :return: a list of float numbers
        """
        sumOut = R.summary(self._model)[0]
        SSR = list(sumOut['Sum Sq'])
        return SSR[:-1]

    def sum_of_squares_res(self):
        """
        Returns the sum of squares of the residuals based on the fitted model,
        retrieved from the R model object.

        :return: a float number

        """
        SSRes = self._model.rx2('residuals')
        SSRes = sum(list(map(lambda x: pow(x, 2), SSRes)))
        return SSRes

    def residual_se(self):
        """
        Returns the residual standard errors from the fitted model, retrieved
        from the R model object

        :return: a float number
        """
        RSE = math.sqrt(AOV.sum_of_squares_res(self) / AOV.df_residual(self))
        return RSE

    def summary(self):
        """
        Prints a cleaned summary output for the fitted aov model like in R.

        :return: string of summary output
        """
        output = "Terms" + "\n" + "              " + "\t"
        maxLength = len("{:e}".format(AOV.sum_of_squares(self)[0])) + 1
        model = AOV.r_model_obj(self)
        headers = list(model.rx2('model').columns)
        for i in range(1, len(headers)):
            output += f"{headers[i]:>{maxLength}}"
        output += f"{'Residuals':>{maxLength}}" + "\n" + "Sum of Squares" \
                  + "\t"
        SSR = AOV.sum_of_squares(self)
        for i in range(len(SSR)):
            output += f"{'{:e}'.format(SSR[i]):>{maxLength}}"
        sos = AOV.sum_of_squares_res(self)
        output += f"{'{:e}'.format(sos):>{maxLength}}" + "\n" + \
                  "Deg. of Freedom" + "\t"
        degFree = AOV.df(self)
        degRes = AOV.df_residual(self)
        for i in range(len(degFree)):
            output += f"{str(int(degFree[i])):>{maxLength}}"
        output += f"{str(degRes):>{maxLength}}" + "\n" + "\n" + \
                  "Residual standard error: "
        RSE = round(AOV.residual_se(self), 1)
        output += str(RSE) + "\n" + "Estimated effects may be unbalanced"
        print(output)
        return output
