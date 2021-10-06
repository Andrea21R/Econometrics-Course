from typing import List
import numpy as np
import pandas as pd
from time import perf_counter

from scipy.optimize import minimize
from scipy.stats import norm


class MaxLik(object):

    def __init__(self,
                 func_vector: 'custom log-likelihood function',
                 data: np.array,
                 initial_guess: np.array,
                 bounds: List[list],
                 constraints: dict = None,
                 delta_incremental: float = 0.00001,
                 method_se_optimization: str = None  # default: information
                 ):

        self.__check_num_pars(pars=initial_guess, data=data)

        # from input
        self.func_vect = func_vector
        self.func_avg = self.__avg_func
        self.data: np.array = data
        self.initial_guess: np.array = initial_guess
        self.bounds: tuple = tuple(tuple(i) for i in bounds)
        self.constraints: dict = constraints
        self.delta_incremental: float = delta_incremental
        self.method_se_optimization: str = method_se_optimization

        # from results
        self.vfunc = None
        self.pars_optimized = None
        self.pars_standard_errors = None
        self.varcov_matrix = None
        self.hessian = None
        self.gradient = None

    def __avg_func(self, parameters, data):

        ll = self.func_vect(parameters, data)
        return -sum(ll) / len(ll)

    @staticmethod
    def __check_num_pars(pars: np.array, data: np.array):
        if len(pars) > len(data):
            raise ValueError("Number of parameters exceeds number of data. You cannot do it!!")

    def __optimizator_boss(self) -> None:

        res = minimize(
            fun=self.func_avg,
            x0=self.initial_guess,
            bounds=self.bounds,
            constraints=self.constraints,
            args=self.data
        )
        self.vfunc = res.fun
        self.pars_optimized = res.x

    def _calc_gradient(self) -> np.array:

        if self.pars_optimized is None:
            self.__optimizator_boss()

        h: float = self.delta_incremental
        pars: np.array = self.pars_optimized

        pos: np.array = np.identity(len(pars))
        gradient: np.array = np.zeros([len(self.data), len(pars)])

        for i in range(len(pars)):
            if pars[i] > 100:
                gradient[:, i] = (self.func_vect(
                    np.multiply(
                        pars,
                        (1 + h * pos[:, i])),
                    self.data
                    )
                 - self.func_vect(pars, self.data)) / (pars[i] * h)
            else:
                gradient[:, i] = (self.func_vect(pars + h * pos[:, i], self.data) - self.func_vect(pars, self.data)) / h

        self.gradient = gradient
        return gradient

    def _calc_hessian(self) -> np.array:

        if self.pars_optimized is None:
            self.__optimizator_boss()

        pars = self.pars_optimized
        h = self.delta_incremental

        pos = np.identity(len(pars))
        H = np.zeros((len(pars), len(pars)))

        for i in range(len(pars)):
            if pars[i] > 100:
                x0P = np.multiply(pars, 1 + (h / 2) * pos[:, i])
                x0N = np.multiply(pars, 1 - (h / 2) * pos[:, i])
                delta_i = pars[i] * h
            else:
                x0P = pars + (h / 2) * pos[:, i]
                x0N = pars - (h / 2) * pos[:, i]
                delta_i = h

            for j in (0, i):

                if pars[j] > 100:
                    x0PP = np.multiply(x0P, 1 + (h / 2) * pos[:, j])
                    x0PN = np.multiply(x0P, 1 - (h / 2) * pos[:, j])
                    x0NP = np.multiply(x0N, 1 + (h / 2) * pos[:, j])
                    x0NN = np.multiply(x0N, 1 - (h / 2) * pos[:, j])
                    fPP = -self.func_vect(x0PP, self.data) / len(self.data)
                    fPN = -self.func_vect(x0PN, self.data) / len(self.data)
                    fNP = -self.func_vect(x0NP, self.data) / len(self.data)
                    fNN = -self.func_vect(x0NN, self.data) / len(self.data)
                    H[i, j] = (sum(fPP) - sum(fPN) - sum(fNP) + sum(fNN)) / (delta_i * h * estimates.x[j])
                    H[j, i] = H[i, j]
                else:
                    x0PP = x0P + (h / 2) * pos[:, j]
                    x0PN = x0P - (h / 2) * pos[:, j]
                    x0NP = x0N + (h / 2) * pos[:, j]
                    x0NN = x0N - (h / 2) * pos[:, j]
                    fPP = -self.func_vect(x0PP, self.data) / len(self.data)
                    fPN = -self.func_vect(x0PN, self.data) / len(self.data)
                    fNP = -self.func_vect(x0NP, self.data) / len(self.data)
                    fNN = -self.func_vect(x0NN, self.data) / len(self.data)
                    H[i, j] = (sum(fPP) - sum(fPN) - sum(fNP) + sum(fNN)) / (h * delta_i)
                    H[j, i] = H[i, j]

        self.hessian = H
        return H

    def _calc_matrix_variance_covariance(self) -> np.array:

        gradient: np.array = self._calc_gradient() if self.gradient is None else self.gradient
        hessian: np.array = self._calc_hessian() if self.hessian is None else self.hessian
        method: str = self.method_se_optimization.lower()

        rows = self.data.shape[0]

        if not method:
            varcov_matrix: np.array = np.linalg.inv(hessian)

        elif method == 'outer':
            varcov_matrix: np.array = np.linalg.inv(
                np.dot(
                    np.transpose(gradient),
                    gradient
                )
            )

        elif 'sandwich':
            outer: np.array = np.dot(np.transpose(gradient), gradient)
            varcov_matrix = np.dot(
                np.dot(
                    np.linalg.inv(
                        hessian * rows
                    ),
                    outer
                ),
                np.linalg.inv(hessian * rows)
            )

        if np.any(np.iscomplex(varcov_matrix)):
            raise Warning("""
            Variance-Covariance matrix returns at least one not real value. Try to use 'outer' method.
            If you've already used 'outer' method, contact Kevyn Stefanelli - Ph.D. Econometrics - Luiss Guido Carli
            """)

        self.varcov_matrix = varcov_matrix
        return varcov_matrix

    def _calc_pars_standard_errors(self) -> np.array:

        if self.varcov_matrix is None:
            self.varcov_matrix: np.array = self._calc_matrix_variance_covariance()

        self.pars_standard_errors: np.array = np.diag(self.varcov_matrix) ** 0.5
        return self.pars_standard_errors

    def _calc_z_test(self) -> np.array:

        if not np.all(self.pars_optimized):
            self.__optimizator_boss()

        return self.pars_optimized / self._calc_pars_standard_errors()

    def _calc_pvalue_test(self):
        return 2 * (1 - norm.cdf(abs(self._calc_z_test())))

    @staticmethod
    def graphic_adjustment(summary: pd.DataFrame) -> pd.DataFrame:

        summary_print: pd.DataFrame = summary.copy(deep=True)

        # ************************** ROUND VALUES **********************************************************************
        summary_print['beta value'] = summary['beta value'].round(2)
        summary_print['SE'] = summary['SE'].round(3)
        summary_print['Z-test'] = summary['Z-test'].round(3)
        summary_print['p-value'] = summary['p-value'].round(3)

        # ************************** Z-TEST AGAINST NORMAL *************************************************************
        summary_print = summary_print.astype(str)
        summary_print['conf'] = np.nan

        for beta in summary.index:
            z = summary.loc[beta, 'Z-test']

            if abs(z) > norm.ppf(0.99):
                summary_print.loc[beta, 'conf'] = '***'
            elif abs(z) > norm.ppf(0.95):
                summary_print.loc[beta, 'conf'] = '**'
            elif abs(z) > norm.ppf(0.9):
                summary_print.loc[beta, 'conf'] = ' *'
            else:
                summary_print.loc[beta, 'conf'] = ' '

        return summary_print.loc[:, ['beta value', 'SE', 'Z-test', 'conf', 'p-value']]

    def get_summary(self, printing: bool = True):

        start = perf_counter()

        if not self.vfunc:
            self.__optimizator_boss()

        rows_name: list = [f'beta{kd}' for kd in range(len(self.pars_optimized))]
        summary: pd.DataFrame = pd.DataFrame(
            columns=['beta value', 'SE', 'Z-test', 'p-value'],
            index=rows_name
        )

        summary['beta value'] = self.pars_optimized
        summary['SE'] = self._calc_pars_standard_errors()
        summary['Z-test'] = self._calc_z_test()
        summary['p-value'] = self._calc_pvalue_test()

        if printing:
            summary_print: pd.DataFrame = self.graphic_adjustment(summary=summary)

            print('===================================================================================================')
            print('======================== MAX LIK OPTIMIZER - by Prof.Carlini et Al. ===============================')
            print('===================================================================================================')

            print(f'\nfunction minimized: {self.func_vect.__name__}')
            print(f'elapsed time: {round(perf_counter() - start, 6)}s')

            print(f'\nnumber of observation: {len(self.data)}')
            print(f'number of parameters: {len(self.pars_optimized)}')
            print(f'function value: {self.vfunc}')
            print(f'\nvariance-covariance method: {self.method_se_optimization}')

            print('\n=================================================================================================')
            print('\n', summary_print)
            print('\n * alpha = 0.1')
            print(' ** alpha = 0.05')
            print(' *** alpha = 0.01')
            print('\n================================= ENJOY ECONOMETRICS GUYS =======================================')

        return summary
