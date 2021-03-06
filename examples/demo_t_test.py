# =======================================
# Created by: Regina-Mae Dominguez
# Created Date: 3/6/2022
# =======================================

import pandas as pd
from simplerpy.t_test import tTest

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

# Test if mean of interest_Rate is equal to 2 in Year 2017
df_2017 = df[df['Year']==2017]
test_One = tTest()
test_One.fit(df_2017['Interest_Rate'], mu=2)
test_One.summary()

# one sample t-test with numpy array
X = df['Unemployment_Rate']
X = X.to_numpy()
Y=[1,2,3,4,5]
test_Two = tTest()
test_Two.fit(Y)
test_Two.summary()

# two sample t-test, assuming equal variances and at 0.90 confidence level
df_2016 = df[df['Year']==2016]
test_Three = tTest()
test_Three.fit(df_2017['Interest_Rate'], df_2016['Interest_Rate'], var_equal=True, conf=0.90)
test_Three.summary()

# extract p-value for two sample test
# performed at default 0.90 confidence level, if significant then there is a difference in means between
# interest rates in 2017 and interest rates in 2016
print('The p-value for this test is ' + str(test_Three.pvalue()))

# extract the mean estimates for this test
print('The mean estimates for this test are ' + str(test_Three.estimate()[0]) + ' and ' + str(test_Three.estimate()[1]))


