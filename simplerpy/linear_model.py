#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: 
# =============================================================================

import pandas as pd
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri

pandas2ri.activate()
R = ro.r

class LM:
    def __init__(self):
        self._model = None

    def fit(self, X_train, y_train, feature_name=None, response_name=None, formula=None):

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
        print("Formula used for fitted model: " + formula)
        self._model = R.lm(formula, data=df)

    def coefficients(self):
        try:
            coeff = self._model
        except:
            raise Exception('Model not fitted.')

    def residuals(self):
        pass

    def standard_error(self):
        pass

    def predict(self, X_train):
        pass

    def p_value(self):
        pass

    def summary(self):
        pass


if __name__ == "__main__":
    model = LM()
    df = pd.read_csv('Sales_sample.csv')
    X_train = df.drop(columns=['LAST_SALE_PRICE'])
    y_train = df['LAST_SALE_PRICE']
    model.fit(X_train, y_train)
    print(R.summary(model._model).rx('coefficients')[0])




