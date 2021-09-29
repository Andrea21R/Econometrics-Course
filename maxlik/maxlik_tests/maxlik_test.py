import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess

from maxlik import MaxLik
from maxlik.likelihood import logLikeAR_1

np.random.seed(12345)
ar2 = np.array([1, 0.5])
ma = np.array([1])
N = 1_000
data = ArmaProcess(ar2, ma).generate_sample(nsample=N)
x0 = np.array([1, 0.4, 1])


bounds = [[-10, 10], [-0.999, 0.999], [0.001, 10]]

boss = MaxLik(
    func_vector=logLikeAR_1,
    data=data,
    initial_guess=x0,
    bounds=bounds,
    method_se_optimization='outer'
)
summary = boss.get_summary(print=True)  # if you set print=False you'll only save df into variable, no printing.
