# ================================================
# Created by: Regina-Mae Dominguez
# Created Date:
# ================================================

from rpy2 import robjects as ro
#importr grabs packages from R
from rpy2.robjects.packages import importr
base = importr('base')
stats = importr('stats')

class tTest:
    def __init__(self) -> None:
        self._model = None
        self._pvalue = None
        self._tvalue = None
        self._df = None
        self._ci = None
        self._estimate = None
        self._stderr = None
        self._alternative = None
        self._method = None
        pass


    def fit(self, data_a, data_b=None, mu=0, var_equal=True):
        R = ro.r
        # one-sample t-test
        dataA = base.as_numeric(data_a)
        if data_a and not data_b:
            #mu is defaulted to 0
            self._model = stats.t_test(dataA, mu=mu)

        # two sample t-test
        if data_a and data_b:
            dataB = base.as_numeric(data_b)
            self._model = stats.t_test(dataA, dataB,**{'var.equal': var_equal})


        #self._pvalue = self._model.rx2('p.value')
        #self._tvalue = self._model.rx2('statistic')
        #self._df = self._model.rx2('parameter')
        #self._ci = self._model.rx2('conf.int')
        #self._estimate = self._model.rx2('estimate')
        #self._stderr = self._model.rx2('stderr')
        #self._alternative = self._model.rx2('alternative')
        #self._method = self._model.rx2('method')


    def pvalue(self):
        return self._pvalue

    def tvalue(self):
        return self._tvalue

    def df(self):
        return self._df

    def ci(self):
        return self._ci

    def estimate(self):
        return self._estimate

    def stderror(self):
        return self._stderr

    def alternative(self):
        return self._alternative

    def method(self):
        return self._method



    def summary(self):
        temp = str(self._model)
        index_of_d = temp.index('d')
        num_of_close = temp.count(')')
        index_of_t = temp[index_of_d + 5:].index('t')

        return temp[:index_of_d] + temp[index_of_d + 5 + index_of_t:]

test = tTest()
x = [1,2,3,4,5]
y = [10,15,32,41,60]
test.fit(x, mu=10)
print(test.summary())


