from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm


class MaxLik(object):

    def __init__(self,
                 func,
                 data: np.array,
                 initial_guess: np.array,
                 bound: List[list],
                 delta_incremental: float = 0.00001,
                 method_se_optimization: str = None  # default: hessian_inv * inv(eye(hessian_inv.shape[0])
                 ):

        self.__check_num_pars(pars=initial_guess, data=data)

        # from input
        self.func = func
        self.data: np.array = data
        self.initial_guess: np.array = initial_guess
        self.bound: tuple = tuple(tuple(i) for i in bound)
        self.delta_incremental: float = delta_incremental
        self.method_se_optimization: str = method_se_optimization

        # from results
        self.vfunc = None
        self.pars_optimized = None
        self.pars_standard_errors = None
        self.varcov_matrix = None
        self.hessian = None
        self.gradient = None

    @staticmethod
    def __check_num_pars(pars: np.array, data: np.array):
        if len(pars) > len(data):
            raise ValueError("Number of parameters exceeds number of data. You cannot do it!!")

    def __optimizator_boss(self):

        res = minimize(
            fun=self.func,
            x0=self.initial_guess,
            bounds=self.bound,
            options={'gtol': 1e-6, 'disp': True}
        )
        self.vfunc = res.fun
        self.pars_optimized = res.x

    def _calc_gradient(self) -> np.array:

        if self.pars_optimized is None:
            self.__optimizator_boss()

        h: float = self.delta_incremental
        pars: np.array = self.pars_optimized

        pos: np.array = np.identity(len(pars))
        gradient: np.array = np.zeros(len(pars))

        for i in range(len(pars)):
            if pars[i] > 1:
                gradient[i] = (self.func(
                    np.multiply(pars, (1 + h * pos[:, i]))) - self.func(pars)) / (pars[i] * h)
            else:
                gradient[i] = (self.func(pars + h * pos[:, i]) - self.func(pars)) / h

        self.gradient = gradient
        return gradient

    def _calc_hessian(self) -> np.array:

        pars = self.pars_optimized
        h = self.delta_incremental

        pos = np.identity(len(pars))
        H = np.zeros((len(pars), len(pars)))

        for i in range(len(pars)):
            if pars[i] > 1:
                x0P = np.multiply(pars, 1 + (h / 2) * pos[:, i])
                x0N = np.multiply(pars, 1 - (h / 2) * pos[:, i])
            else:
                x0P = pars + (h / 2) * pos[:, i]
                x0N = pars - (h / 2) * pos[:, i]
            c = range(i + 1)
            for j in c:
                if pars[j] > 1:
                    x0PP = np.multiply(x0P, 1 + (h / 2) * pos[:, j])
                    x0PN = np.multiply(x0P, 1 - (h / 2) * pos[:, j])
                    x0NP = np.multiply(x0N, 1 + (h / 2) * pos[:, j])
                    x0NN = np.multiply(x0N, 1 - (h / 2) * pos[:, j])
                    H[i, j] = \
                        (self.func(x0PP) - self.func(x0PN) - self.func(x0NP) + self.func(x0NN)) / (h * pars[j])
                    H[j, i] = H[i, j]
                else:
                    x0PP = x0P + (h / 2) * pos[:, j]
                    x0PN = x0P - (h / 2) * pos[:, j]
                    x0NP = x0N + (h / 2) * pos[:, j]
                    x0NN = x0N - (h / 2) * pos[:, j]
                    H[i, j] = (self.func(x0PP) - self.func(x0PN) - self.func(x0NP) + self.func(x0NN)) / h
                    H[j, i] = H[i, j]

        self.hessian = H
        return H

    def _calc_matrix_variance_covariance(self) -> np.array:

        gradient: np.array = self._calc_gradient() if self.gradient is None else self.gradient
        hessian: np.array = self._calc_hessian() if self.hessian is None else self.hessian
        method: str = self.method_se_optimization.lower()

        rows = self.data.shape[0]

        if not method:
            varcov_matrix: np.array = -np.linalg.inv(hessian * rows)

        elif method == 'outer':
            varcov_matrix = np.linalg.inv(np.outer(gradient, gradient))

        elif 'sandwich':
            varcov_matrix = np.dot(
                np.dot(-np.linalg.inv(hessian * rows),
                       np.outer(gradient, gradient)),
                np.linalg.inv(hessian * rows)
            )

        if np.any(np.iscomplex(varcov_matrix)):
            raise Warning("""
            Variance-Covariance matrix returns at least one not real value. Try to use 'outer' method.
            If you've already used 'outer' method, contact Kevyn Stefanelli - Ph.D. Econometrics - Luiss Guido Carli
            """)

        self.varcov_matrix = varcov_matrix
        return varcov_matrix

    def _calc_pars_standard_errors(self):

        varcov_matrix: np.array = self._calc_matrix_variance_covariance()

        self.pars_standard_errors: np.array = np.diag(varcov_matrix) ** (1 / 2)
        return self.pars_standard_errors

    def _calc_z_test(self) -> np.array:

        self.__optimizator_boss()

        pars = self.pars_optimized
        se = self._calc_pars_standard_errors()

        return pars / se

    def _calc_pvalue_test(self):
        z_test: np.array = self._calc_z_test()
        pvalues: np.array = 2 * (1 - norm.cdf(abs(z_test)))

        return pvalues

    def get_summary(self):

        if not self.vfunc:
            self.__optimizator_boss()

        se: np.array = self._calc_pars_standard_errors()
        ztest: np.array = self._calc_z_test()
        pvalue: np.array = self._calc_pvalue_test()

        rows_name: list = [f'beta{par}' for par in self.pars_optimized]
        summary: pd.DataFrame = pd.DataFrame(
            columns=['beta value', 'SE', 'Z-test', 'p-value'],
            index=rows_name
        )

        summary['beta value'] = self.pars_optimized
        summary['SE'] = se
        summary['Z-test'] = ztest
        summary['p-value'] = pvalue

        print('======================================================================================================')
        print('======================== MAX LIK OPTIMIZER - by Prof.Carlini et Al. ==================================')
        print('======================================================================================================')

        print(f'\nnumber of observation: {len(self.data)}')
        print(f'number of parameters: {len(self.pars_optimized)}')
        print(f'function value: {self.vfunc}')
        print(f'\nvariance-covariance method: {self.method_se_optimization}')

        print('\n=====================================================================================================')
        print('\n', summary)
        print('\n ================================= ENJOY ECONOMETRICS GUYS ==========================================')

        return summary

if __name__ == "__main__":
    import numpy as np
    from statsmodels.tsa.arima_process import ArmaProcess

    np.random.seed(12345)
    ar2 = np.array([1, 0.5])
    ma = np.array([1])
    N = 100
    sim = ArmaProcess(ar2, ma).generate_sample(nsample=N)

    def negloglikeAR(xi):
        S = np.sum((sim[1:N - 1] - xi[0] * np.ones(N - 2) - xi[1] * sim[0:N - 2]) ** 2)
        return N / 2 * np.log(2 * np.pi * xi[2]) + S / (2 * xi[2])

    x0 = np.array([0.5, 1, 1])
    bounds = [[-10, 10], [-10, 10], [0.001, 10]]

    boss = MaxLik(func=negloglikeAR, data=sim, initial_guess=x0, bound=bounds, method_se_optimization='sandwich')
    boss.get_summary()

