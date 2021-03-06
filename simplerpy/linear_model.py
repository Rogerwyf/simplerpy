#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: 3/1/2022
# =============================================================================

"""
The class 'LM' contains an ordinary linear model object (lm) from R through the
bridge package rpy2. This class mimics how most machine learning models in
Python work and provide extra information retrieved from the R object.
"""

import logging

import pandas as pd
from rpy2 import robjects as ro
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
from rpy2.robjects import pandas2ri

rpy2_logger.setLevel(logging.ERROR)
pandas2ri.activate()
R = ro.r


class LM:
    def __init__(self):
        """
        Initialize the LM object, fit() must be called before calling all other
        methods of this object.
        """

        self._model = None

    def fit(self, X, y, feature_name=None, response_name=None, formula=None,
            verbose=1):
        """
        Fits the linear model with input training features X_train and target
        y_train with lm() in R through the package rpy2.

        :param X: feature vector, array-like (n_samples, n_features) or
        Pandas Dataframe
        :param y: target vector, array-like (n_samples, 1) or Pandas Series or
        Dataframe
        :param feature_name: names of features, used only when X_train is
        array-like all feature names in the final model will be "f_num"
        by default
        :param response_name: name of the target, used when y_train is
        array-like, "y" by default.
        :param formula: formula used for lm() if specified
        :param verbose: prints out formula used for lm() if value equals to 1,
        silenced if 0

        :return: None, assign the R model object to self._model
        """

        # check if input target vector has a name
        if isinstance(y, pd.Series):
            res_name = y.name
        elif response_name:
            res_name = response_name
        else:
            res_name = 'y'

        # check if input is a PD Dataframe and prepare data for lm()
        if type(X) == pd.DataFrame:
            col_names = X.columns.values.tolist()
            df = X.copy()

        else:
            if feature_name:
                col_names = feature_name

            else:
                col_names = []
                for i in range(len(X[0])):
                    col_names.append('f' + str(i + 1))
            df = pd.DataFrame(X, columns=col_names)

        df[res_name] = y

        # check if formula for lm() is specified, all features are used if not
        if not formula:
            formula = res_name + " ~ "
            for i in range(len(col_names)):
                formula += col_names[i]
                if i != len(col_names) - 1:
                    formula += " + "

        self._model = R.lm(formula, data=df)

        # check if printout is needed
        if verbose:
            print("Formula used for fitted model: " + formula)

    def r_model_obj(self):
        """
        Returns the fitted R model object.

        :return: a R model object
        """
        if self._model:
            return self._model
        else:
            raise ValueError('model not fitted')

    def coefficient(self):
        """
        Returns the coefficients of the fitted linear model, retrieved from
        the R model object.

        :return: a list of float numbers
        """
        if self._model:
            coeff = [result[0] for result in
                     R.summary(self._model).rx('coefficients')[0]]
            return coeff
        else:
            raise ValueError('model not fitted')

    def df_residual(self):
        """
        Returns the degree of freedom on residuals of the fitted model,
        retrieved from the R model object.

        :return: a float number
        """
        if self._model:
            df = self._model.rx('df.residual')[0][0]
            return df
        else:
            raise ValueError('model not fitted')

    def residuals(self):
        """
        Returns the residuals on training data based on the fitted model,
        retrieved from the R model object.

        :return: a list of float numbers
        """
        if self._model:
            resid = self._model.rx('residuals')[0]
            return resid.tolist()
        else:
            raise ValueError('model not fitted')

    def standard_error(self):
        """
        Returns the standard errors on features of the fitted model, retrieved
        from the R model object.

        :return: a list of float numbers
        """
        if self._model:
            se = [result[1] for result in
                  R.summary(self._model).rx('coefficients')[0]]
            return se
        else:
            raise ValueError('model not fitted')

    def predict(self, X_test):
        """
        Takes feature vectors with the same shape as training data and
        returns predictions based on the fitted model.

        :param X_test: feature vectors, each must be of the same shape of the
        training data
        :return: a list of float numbers as predictions
        """

        if self._model:
            return R.predict(self._model, X_test)
        else:
            raise ValueError('model not fitted')

    def test_stats(self):
        """
        Returns the test statistics for each coefficient of the model,
        retrieved from the R model object.

        :return: a list of float numbers
        """
        if self._model:
            t_value = [result[2] for result in
                       R.summary(self._model).rx('coefficients')[0]]
            return t_value
        else:
            raise ValueError('model not fitted')

    def p_value(self):
        """
        Returns the p-value of test of siginificance on each coefficient of the
        model, retrieved from the R model object.

        :return: a list of float numbers
        """
        if self._model:
            p_value = [result[3] for result in
                       R.summary(self._model).rx('coefficients')[0]]
            return p_value
        else:
            raise ValueError('model not fitted')

    def fitted_values(self):
        """
        Returns the fitted values on training data based on the fitted model,
        retrieved from the R model object

        :return: a list of float numbers
        """
        if self._model:
            fv = self._model.rx('fitted.values')[0]
            return fv.tolist()
        else:
            raise ValueError('model not fitted')

    def r_squared(self):
        """
        Returns R squared of the fitted model, retrieved from the R model
        object

        :return: a float number
        """
        if self._model:
            rs = R.summary(self._model).rx('r.squared')[0][0]
            return rs
        else:
            raise ValueError('model not fitted')

    def adj_r_squared(self):
        """
        Returns adjusted R squared of the fitted model, retrieved from the R
        model object

        :return: a float number
        """
        if self._model:
            adj_rs = R.summary(self._model).rx('adj.r.squared')[0][0]
            return adj_rs
        else:
            raise ValueError('model not fitted')

    def f_statistic(self):
        """
        Returns f statistic and its degree of freedom of the fitted model in
        the format of (f-stat, df1, df2), retrieved from the R model object

        :return: a list of float numbers
        """
        if self._model:
            fstats = R.summary(self._model).rx('fstatistic')[0]
            return fstats.tolist()
        else:
            raise ValueError('model not fitted')

    def f_test_pvalue(self):
        """
        Returns the p-value of F test on the fitted model, calculated based on
        information retrieved from the R model object using pf() function from
        R through rpy2

        :return: a float number
        """
        if self._model:
            test_info = self.f_statistic()
            f_stat = test_info[0]
            df_num = test_info[1]
            df_denom = test_info[2]
            return R.pf(f_stat, df_num, df_denom, **{'lower.tail': False})[0]
        else:
            raise ValueError('model not fitted')

    def residual_se(self):
        """
        Returns the residual standard errors from the fitted model, retrieved
        from the R model object

        :return: a float number
        """
        if self._model:
            return R.summary(self._model).rx('sigma')[0][0]
        else:
            raise ValueError('model not fitted')

    def summary(self):
        """
        Print a summary for the fitted model that mimics the printout from
        summary() function in R.

        :return: String contains the output summary
        """
        if self._model:
            output = ""
            output += str(R.summary(self._model).rx('coefficients'))
            output += f'Residual standard error: ' \
                      f'{round(self.residual_se(), 6)} ' \
                      f'on {self.df_residual()} degrees of freedom\n'
            output += f'Mutiple R-squared: {round(self.r_squared() , 6)}, ' \
                      f'Adjusted R-squared: {round(self.adj_r_squared(), 6)}\n'
            output += f'F-statistic: {round(self.f_statistic()[0], 6)} on ' \
                      f'{self.f_statistic()[1]} and {self.f_statistic()[2]} ' \
                      f'DF with p-value: {round(self.f_test_pvalue(), 6)}'

            print(output)

            return output
        else:
            raise ValueError('model not fitted')
