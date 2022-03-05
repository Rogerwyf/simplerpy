#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: 3/3/2022
# =============================================================================
import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()
R = ro.r

class LM:
    def __init__(self):
        self._model = None

    def fit(self, X_train, y_train, feature_name=None, response_name=None, formula=None,
            verbose=1):

        if type(y_train) == pd.core.series.Series:
            res_name = y_train.name
        elif response_name:
            res_name = response_name
        else:
            res_name = 'y'

        if type(X_train) == pd.DataFrame:
            col_names = X_train.columns.values.tolist()
            df = X_train

        else:
            if feature_name:
                col_names = feature_name

            else:
                col_names = []
                for i in range(len(X_train[0])):
                    col_names.append('f' + str(i + 1))
            df = pd.DataFrame(X_train, columns=col_names)

        df[res_name] = y_train

        if not formula:
            formula = res_name + " ~ "
            for i in range(len(col_names)):
                formula += col_names[i]
                if i != len(col_names) - 1:
                    formula += " + "

        if verbose:
            print("Formula used for fitted model: " + formula)
        self._model = R.lm(formula, data=df)

    def r_model_obj(self):
        if self._model:
            return self._model
        else:
            raise ValueError('model not fitted')

    def coefficient(self):
        if self._model:
            coeff = [result[0] for result in R.summary(self._model).rx('coefficients')[0]]
            return coeff
        else:
            raise ValueError('model not fitted')

    def df_residual(self):
        if self._model:
            df = self._model.rx('df.residual')[0][0]
            return df
        else:
            raise ValueError('model not fitted')

    def residuals(self):
        if self._model:
            resid = self._model.rx('residuals')[0]
            return resid
        else:
            raise ValueError('model not fitted')

    def standard_error(self):
        if self._model:
            se = [result[1] for result in R.summary(self._model).rx('coefficients')[0]]
            return se
        else:
            raise ValueError('model not fitted')

    def predict(self, X_test):
        if self._model:
            return R.predict(self._model, X_test)
        else:
            raise ValueError('model not fitted')

    def test_stats(self):
        if self._model:
            t_value = [result[2] for result in R.summary(self._model).rx('coefficients')[0]]
            return t_value
        else:
            raise ValueError('model not fitted')

    def p_value(self):
        if self._model:
            p_value = [result[3] for result in R.summary(self._model).rx('coefficients')[0]]
            return p_value
        else:
            raise ValueError('model not fitted')

    def fitted_values(self):
        if self._model:
            fv = self._model.rx('fitted.values')[0]
            return fv
        else:
            raise ValueError('model not fitted')

    def r_squared(self):
        if self._model:
            rs = R.summary(self._model).rx('r.squared')[0][0]
            return rs
        else:
            raise ValueError('model not fitted')

    def adj_r_squared(self):
        if self._model:
            adj_rs = R.summary(self._model).rx('adj.r.squared')[0][0]
            return adj_rs
        else:
            raise ValueError('model not fitted')

    def f_statistic(self):
        if self._model:
            adj_rs = R.summary(self._model).rx('fstatistic')[0]
            return adj_rs
        else:
            raise ValueError('model not fitted')

    def f_test_pvalue(self):
        if self._model:
            test_info = self.f_statistic()
            f_stat = test_info[0]
            df_num = test_info[1]
            df_denom = test_info[2]
            return R.pf(f_stat, df_num, df_denom, **{'lower.tail': False})[0]
        else:
            raise ValueError('model not fitted')

    def residual_se(self):
        if self._model:
            return R.summary(self._model).rx('sigma')[0][0]
        else:
            raise ValueError('model not fitted')

    def summary(self):
        print(R.summary(self._model).rx('coefficients'))
        print(f'Residual standard error: {round(self.residual_se(), 6)} on {self.df_residual()} '
              f'degrees of freedom')
        print(f'Mutiple R-squared: {round(self.r_squared() , 6)}, Adjusted R-squared:'
              f' {round(self.adj_r_squared(), 6)}')
        print(f'F-statistic: {round(self.f_statistic()[0], 6)} on {self.f_statistic()[1]} and '
              f'{self.f_statistic()[2]} DF with p-value: {round(self.f_test_pvalue(), 6)}')