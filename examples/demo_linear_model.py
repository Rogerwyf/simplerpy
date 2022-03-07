#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By  : Roger Wang
# Created Date: 3/4/2022
# =============================================================================

"""
A demo for using class LM in the package simplerpy
"""

import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from simplerpy.linear_model import LM

# Demo data
Stock_Market = {
    'Year': [2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2016, 2016,
             2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016],
    'Month': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'Interest_Rate': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75,
                      1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75],
    'Unemployment_Rate': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8,
                          6.1, 6.2, 6.1, 6.1, 6.1, 5.9, 6.2, 6.2, 6.1],
    'Stock_Index_Price': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075,
                          1047, 965, 943, 958, 971, 949, 884, 866, 876, 822, 704, 719]
    }

df = pd.DataFrame(Stock_Market, columns=['Year', 'Month', 'Interest_Rate', 'Unemployment_Rate',
                                         'Stock_Index_Price'])
X = df.drop(columns=['Stock_Index_Price'])
y = df['Stock_Index_Price']

# Train an OLS model like how it is done in R
model = LM()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model.fit(X_train, y_train)

# Get a summary of the fitted model
model.summary()

# Get relevant information of the fitted model, see more in linear_model.py
print(f'Coefficients of the features are: {model.coefficient()}')
print(f'Standard errors of the features are: {model.standard_error()}')
print(f'p-values of test of significance on the features are: {model.p_value()}')
print(f'r-squared of the model is: {model.r_squared()}')
print(f'f statistic of the model is {model.f_statistic()[0]} with degrees of freedom of '
      f'{model.f_statistic()[1]} and {model.f_statistic()[2]}')
print(f'The resiual standard error based on the fitted model is {model.residual_se()}')

# Make predictions and calculate metrics
y_pred = model.predict(X_test)
print(f"Mean Squared Error on testing data: {mean_squared_error(y_test, y_pred)}\n")

# Specify custom formula for lm
model2 = LM()
model2.fit(X_train, y_train, formula="Stock_Index_Price ~ Year + Month + Unemployment_Rate")
model2.summary()

# LM() works with list of lists or numpy arrays
model3 = LM()
X = X.to_numpy() # convert pandas series to numpy arrays
y = y.to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model3.fit(X_train, y_train)
model3.summary()