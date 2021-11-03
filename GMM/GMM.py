from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
from time import perf_counter


class GMM(object):

    def __init__(self,
                 data: np.array,
                 func_h: 'custom function',
                 func_g_hat: 'custom function',
                 initial_guess: np.array,
                 GMM_lags: int,
                 bounds: List[list] = None,
                 max_iter: int = 10_000,
                 tolerance_optmz1: float = 0.000000001,
                 tolerance_optmz2: float = 0.000000001,
                 constraints: dict = None
                 ):
        # ================================ USER INPUT ==================================================================
        self.data: np.array = data
        self.initial_guess: np.array = initial_guess
        self.func_h = func_h
        self.func_g_hat = func_g_hat
        self.GMM_lags: int = GMM_lags
        self.bounds: List[list] = bounds
        self.max_iter: int = max_iter
        self.tolerance_optmz1 = tolerance_optmz1
        self.tolerance_optmz2 = tolerance_optmz2
        self.k: int = max(func_g_hat(pars_vect=initial_guess, data=data).shape)
        self.constraints: dict = constraints

        # ================================ CLASS OUTPUT ================================================================
        self.func_val_optimized = None
        self.pars_optimized = None
        self.omega = None

    def __manage_g_hat_shape(self, g_hat: np.array):
        # shape check (result_g_hat must be a 1D vector)
        if g_hat.shape != (self.k,):
            g_hat = g_hat.reshape(self.k, )
        return g_hat

    def _obj_func(self, pars_vect: np.array, omega: np.array):

        g_hat: np.array = self.func_g_hat(pars_vect=pars_vect, data=self.data)
        # shape check (result_g_hat must be a 1D vector)
        if g_hat.shape != (self.k,):
            g_hat = g_hat.reshape(self.k,)

        inv_omega = np.linalg.inv(omega)

        # givend the 1D dimension of g_hat, the transpose is not necessary
        return g_hat @ inv_omega @ g_hat.transpose()

    def _calc_gradient(self):

        if not self.func_val_optimized:
            self._optimizer_boss()

        h = 0.00001
        n = max(self.initial_guess.shape)
        pos = np.identity(n)
        grad = np.zeros((self.k, n))

        func_g_hat = self.func_g_hat
        data = self.data
        beta = self.pars_optimized

        for i in range(n):
            x: np.array = func_g_hat(np.multiply(beta, (1 + h * pos[:, i])), data)
            xx: np.array = func_g_hat(beta, data)

            grad[:, i] = ((x - xx) / (beta[i] * h))[:, i]

        return grad

    def _optimizer_boss(self):

        # input preparation
        t: int = len(self.data)
        init_func: int = 10_000
        x0 = self.initial_guess

        omega: np.array = np.identity(self.k)

        count = 0
        while count <= self.max_iter:
            res = minimize(
                fun=self._obj_func,
                x0=x0,
                args=omega,
                bounds=self.bounds,
                options={'gtol': 1e-6, 'disp': False}
            )

            if abs(res.fun - init_func) < self.tolerance_optmz1 and \
               abs(max(res.x - x0)) < self.tolerance_optmz2:
                break

            x0 = res.x
            init_func = res.fun

            H: np.array = self.func_h(pars_vect=res.x, data=self.data)
            mH: np.array = np.zeros((1, self.k))

            for j in range(0, self.k):
                mH[0, j] = np.mean(H[:, j])

            H = H - np.kron(mH, np.ones((H.shape[0], 1)))
            omega: np.array = (H.transpose() @ H) / t

            # if GMM_lags == 0 the following for loop doesn't run
            for j in range(0, self.GMM_lags):
                gamma: np.array = (H[j + 1: t, :].transpose() @ H[0: t-j-1, :]) / t
                omega = omega + (1 - (j + 1)/(self.GMM_lags + 1)) * (gamma + gamma.transpose())

            count += 1

        self.func_val_optimized = res.fun
        self.pars_optimized = res.x
        self.omega = omega

    def _calc_inference_stats(self):

        if not self.pars_optimized:
            self._optimizer_boss()

        grad: np.array = self._calc_gradient()
        t = len(self.data)

        covar_matrix: np.array = (1/t) * np.linalg.inv(grad.transpose() @ np.linalg.inv(self.omega) @ grad)

        std_error: np.array = np.sqrt(np.diag(covar_matrix))
        t_test: np.array = self.pars_optimized / std_error
        p_value: np.array = 2 * (1 - norm.cdf(t_test))

        return std_error, t_test, p_value

    def _calc_SarganHansen_test(self):

        if not self.func_val_optimized:
            self._optimizer_boss()

        t: int = len(self.data)
        df: int = self.k - self.pars_optimized.shape[0]

        stat_test: float = t * self.func_val_optimized
        p_test: float = 1 - chi2.cdf(stat_test, df)

        return stat_test, p_test

    def get_summary(self, printing: bool = True):

        start = perf_counter()

        if not self.func_val_optimized:
            self._optimizer_boss()
        std_error, t_test, p_value = self._calc_inference_stats()

        if self.k > max(self.initial_guess.shape):
            stat_test, p_test = self._calc_SarganHansen_test()
        else:
            stat_test, p_test = "n.a.", "n.a."

        rows = [f"pars{i}" for i in range(1, len(self.pars_optimized) + 1)]
        summary = pd.DataFrame(index=rows, columns=['Pars Value', 'SE', 'T-test', 'P-Value'])
        summary['Pars Value'] = self.pars_optimized
        summary['SE'] = std_error
        summary['T-test'] = t_test
        summary['P-Value'] = p_value

        if printing:
            print('===================================================================================================')
            print('======================== GMM OPTIMIZER - by Prof.Carlini et Al. ===================================')
            print('===================================================================================================')

            print(f'\nelapsed time: {round(perf_counter() - start, 6)}s')

            print(f'\n# of observation: {len(self.data)}')
            print(f'# estimated pars: {max(self.initial_guess.shape)}')
            print(f'# degree of freedom: {self.k - self.pars_optimized.shape[0]}')
            print(f'# of orthogonality condition: {self.k}')
            print(f'Value of the objective function: {self.func_val_optimized}')
            print(f'Test of over-identification of restrictions: {round(stat_test, 6)}')
            print(f'P-Value of over-identification: {str(round(p_test, 4) * 100)}%')

            print('\n=================================================================================================')
            print(f'\n{summary}')
            print('\n================================= ENJOY ECONOMETRICS GUYS =======================================')

        return summary

if __name__ == "__main__":

    from scipy.stats import t
    import random

    # ================= RANDOM SAMPLE BY T-DISTRIBUTION ================================================================
    np.random.seed(100)
    data = t.rvs(11, size=1000)

    # ================= DEFINE FUNCTION ================================================================================

    def func_moment_cond(pars_vect: np.array, data: np.array):
        h_1 = data ** 2 - pars_vect / (pars_vect - 2)
        h_2 = data ** 4 - (3 * pars_vect ** 2) / ((pars_vect - 2) * (pars_vect - 4))
        moment_conditions = np.vstack((h_1, h_2))
        moment_conditions = moment_conditions.transpose()
        return moment_conditions


    def func_g_hat(pars_vect, data):
        T = len(data)
        h_1 = (1 / T) * sum(data ** 2) - pars_vect / (pars_vect - 2)
        h_2 = (1 / T) * sum(data ** 4) - (3 * pars_vect ** 2) / ((pars_vect - 2) * (pars_vect - 4))
        moment_conditions = np.vstack((h_1, h_2))
        return moment_conditions

    # ================= MAKE OPTIMIZATION ==============================================================================
    GMM = GMM(
        data=data,
        func_h=func_moment_cond,
        func_g_hat=func_g_hat,
        initial_guess=np.array([4.5]),
        GMM_lags=0,
        bounds=[[4.00001, None]],
        max_iter=10_000,
        tolerance_optmz1=0.00000000000001,
        tolerance_optmz2=0.00000000000001,
        constraints=None  # we don't have any constraints for this case
    )

    GMM.get_summary(printing=True)



