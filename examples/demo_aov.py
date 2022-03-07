#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Created By: Michelle Hsieh
# Created Date: 3/6/2022
# =============================================================================

"""
A demo for using class AOV in the package simplerpy
"""

import pandas as pd
from simplerpy.aov import AOV

Stock_Market = {
    'Year': [2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2017, 2016, 2016,
             2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016, 2016],
    'Month': [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    'IR': [2.75, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.25, 2.25, 2.25, 2, 2, 2, 1.75, 1.75,
                      1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75, 1.75],
    'UR': [5.3, 5.3, 5.3, 5.3, 5.4, 5.6, 5.5, 5.5, 5.5, 5.6, 5.7, 5.9, 6, 5.9, 5.8,
                          6.1, 6.2, 6.1, 6.1, 6.1, 5.9, 6.2, 6.2, 6.1],
    'Stock_Index_Price': [1464, 1394, 1357, 1293, 1256, 1254, 1234, 1195, 1159, 1167, 1130, 1075,
                          1047, 965, 943, 958, 971, 949, 884, 866, 876, 822, 704, 719]
    }

df = pd.DataFrame(Stock_Market, columns=['Year', 'Month', 'IR', 'UR',
                                         'Stock_Index_Price'])
X = df.drop(columns=['Stock_Index_Price'])
y = df['Stock_Index_Price']


#fits an AOV model and displays the summary

model=AOV() 
model.fit("Stock_Index_Price~Year+Month+IR+UR", df) 
model.summary()

#Specify custom formulas for aov

model2=AOV()
model2.fit("Stock_Index_Price~Year+Month", df)
model2.summary()

model3=AOV()
model3.fit("Stock_Index_Price~UR", df)
model3.summary()


# Extract parts of AOV summary output

print(model.sum_of_squares()) # returns list of sum of square values
print(model2.df()) # returns list of degrees of freedom
