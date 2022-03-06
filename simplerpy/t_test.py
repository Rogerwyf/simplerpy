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


    def pvalue(self):
        if self._model:
            return self._model.rx2('p.value')[0]
        else:
            raise ValueError('Model not fitted')

    def tvalue(self):
        if self._model:
            return self._model.rx2('statistic')[0]
        else:
            raise ValueError('Model not fitted')

    def df(self):
        if self._model:
            return self._model.rx2('parameter')[0]
        else:
            raise ValueError('Model not fitted')

    def ci(self):
        if self._model:
            return self._model.rx2('conf.int')[0]
        else:
            raise ValueError('Model not fitted')

    def estimate(self):
        if self._model:
            return self._model.rx2('estimate')[0]
        else:
            raise ValueError('Model not fitted')

    def stderror(self):
        if self._model:
            return self._model.rx2('stderr')[0]
        else:
            raise ValueError('Model not fitted')

    def alternative(self):
        if self._model:
            return self._model.rx2('alternative')[0]
        else:
            raise ValueError('Model not fitted')

    def method(self):
        if self._model:
            return self._model.rx2('method')[0]
        else:
            raise ValueError('Model not fitted')



    def summary(self):
        temp = str(self._model)
        index_of_d = temp.index('d')
        num_of_close = temp.count(')')
        index_of_t = temp[index_of_d + 5:].index('t')

        return temp[:index_of_d] + temp[index_of_d + 5 + index_of_t:]

if __name__=="__main__":
    test = tTest()
    x = [1,2,3,4,5]
    y = [10,15,32,41,60]
    test.fit(x, mu=10)
    #print(test.summary())
    print(test.pvalue())


