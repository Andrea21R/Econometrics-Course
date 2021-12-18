import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import ArmaProcess

from maxlik import MaxLik
from maxlik.likelihood import ar1_loglike, garch_loglik

# ========================================= AR(1) TEST =================================================================
np.random.seed(12345)
ar2 = np.array([1, 0.5])
ma = np.array([1])
N = 1_000
data = ArmaProcess(ar2, ma).generate_sample(nsample=N)
x0 = np.array([1, 0.4, 1])

bounds = [[-10, 10], [-0.999, 0.999], [0.001, 10]]

boss = MaxLik(
    func_vector=ar1_loglike,
    data=data,
    initial_guess=x0,
    bounds=bounds,
    method_se_optimization='outer'
)
summary = boss.get_summary(printing=True)  # if you set printing=False you'll only save df into variable, no printing.

# ========================================= GARCH(1) TEST ==============================================================
data = np.array(pd.read_excel("~\PycharmProjects\Econometrics-Course\maxlik\maxlik_tests\GARCH_simulation.xlsx"))
bounds = [[-10, 10], [0.000009, 10], [0.000009, 10], [0.000009, 10]]
cons = ({'type': 'ineq', 'fun': lambda p: -(p[2] + p[3]) + 1})
x0 = np.array([1, 0.4, 1, 1])

boss = MaxLik(
    func_vector=garch_loglik,
    data=data,
    initial_guess=x0,
    bounds=bounds,
    constraints=cons,
    method_se_optimization='outer'
)
print(' \n')
summary = boss.get_summary(printing=True)
#
