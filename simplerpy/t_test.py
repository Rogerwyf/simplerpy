# ================================================
# Created by: Regina-Mae Dominguez
# Created Date:
# ================================================


#importr grabs packages from R
from rpy2.robjects.packages import importr
base = importr('base')
stats = importr('stats')
"""
The class "tTest" performs one or two sample t-tests from R through the bridge package rpy2.
This class mimics Python commands and outputs while providing extra information
for the rest retrieved from the R object.  
"""
class tTest:
    def __init__(self):
        """
        Initialize the tTest object, fit() must be called before all other methods
        """
        self._model = None



    def fit(self, data_a, data_b=None, mu=0, var_equal=True, conf=0.95, paired=False, alternative= "two.sided"):
        """
        Run the ttest with different features with stats.t_test in R
        through the package rpy2.

        :param data_a: vector
        :param data_b: vector, used only for two sample test
        :param mu: numeric value, default to 0
        :param var_equal: boolean, true for equal or false for unequal variances test
        :param conf: float numeric, confidence level for interval/testing
        :param paired: boolean, true for paired t-test, default to false
        :param alternative: c('two.sided', 'less', 'greater'), type of test to perform

        :return: None, assign to self._model
        """
        if type(data_a) == pd.core.series.Series:
            dataA = data_a
        else:
            dataA = base.as_numeric(data_a)
        # one-sample t-test
        if data_a and not data_b:
            #mu is defaulted to 0
            self._model = stats.t_test(dataA, mu=mu, **{'conf.level': conf,
                                                        'alternative': alternative})

        # two sample t-test
        if data_a and data_b:
            if type(data_b) == pd.core.series.Series:
                dataB = data_b
            else:
                dataB = base.as_numeric(data_b)
            self._model = stats.t_test(dataA, dataB,**{'var.equal': var_equal,
                                                       'conf.level': conf,
                                                       'paired': paired,
                                                       'alternative': alternative})


    def pvalue(self):
        """
        Returns p-value obtained from test

        return: a float value
        """
        if self._model:
            return self._model.rx2('p.value')[0]
        else:
            raise ValueError('Model not fitted')

    def tvalue(self):
        """
        Returns the test statistic

        return: a float value
        """
        if self._model:
            return self._model.rx2('statistic')[0]
        else:
            raise ValueError('Model not fitted')

    def df(self):
        """
        Returns the degrees of freedom of test

        return: numeric value
        """
        if self._model:
            return self._model.rx2('parameter')[0]
        else:
            raise ValueError('Model not fitted')

    def ci(self):
        """
        Returns the confidence interval of test

        return: float vector
        """
        if self._model:
            return self._model.rx2('conf.int')[:2]
        else:
            raise ValueError('Model not fitted')

    def estimate(self):
        """
        Returns the estimated mean or difference in means

        return: float value or list of float values
        """
        if self._model:
            return self._model.rx2('estimate')
        else:
            raise ValueError('Model not fitted')

    def stderror(self):
        """
        Returns the standard error of the mean(difference)

        return: float value
        """
        if self._model:
            return self._model.rx2('stderr')[0]
        else:
            raise ValueError('Model not fitted')

    def alternative(self):
        """
        Returns the alternative hypothesis

        return: string
        """
        if self._model:
            return self._model.rx2('alternative')[0]
        else:
            raise ValueError('Model not fitted')

    def method(self):
        """
        Returns type of t-test performed

        return: string
        """
        if self._model:
            return self._model.rx2('method')[0]
        else:
            raise ValueError('Model not fitted')


    def summary(self):
        """
        Prints summary of ttest

        return: none
        """
        temp = str(self._model)
        index_of_d = temp.index('d')
        num_of_close = temp.count(')')
        index_of_t = temp[index_of_d + 5:].index('t')

        print(temp[:index_of_d] + temp[index_of_d + 5 + index_of_t:])

#if __name__=="__main__":
 #   test = tTest()
  #  x = [1,2,3,4,5]
  #  y = [10,15,32,41,60]
   # test.fit(x, y)
   # test.summary()
    #print(test.tvalue())
   # print(test.method())
   # print(type(test.method()))
    #test2 = tTest()
    #test2.fit(x, alternative='less', mu=10)
    #print(test2.method())
    #print(type(test2.method()))
    #test2.summary()


