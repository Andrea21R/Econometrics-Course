from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import chi2
from time import perf_counter


class GMM(object):
    """
    WARNING.1: func_vector input (che è una funzione) deve avere come input due parametri che DEVONO ESSERE
               NECESSARIAMENTE CHIAMATI: pars_vect, data
    WARNING.2: func_vector, parameter "pars_vect" DEVE NECESSARIAMENTE ESSERE UN NUMPY.ARRAY 1D (ex. shape(n,))
    """
    def __init__(self,
                 data: np.array,
                 func_h: 'custom function',
                 num_conditions: int,
                 initial_guess: np.array,
                 GMM_lags: int,
                 bounds: List[list] = None,
                 max_iter: int = 10_000,
                 tolerance_optmz1: float = 0.000000001,
                 tolerance_optmz2: float = 0.000000001,
                 constraints: dict = None
                 ):
        # ================================ USER INPUT ==================================================================
        self.k = num_conditions
        self.initial_guess: np.array = initial_guess
        self.__check_missspecification()
        self.data: np.array = data
        self.func_h = func_h
        self.__check_func_h_pars()
        self.GMM_lags: int = GMM_lags
        self.bounds: List[list] = bounds
        self.max_iter: int = max_iter
        self.tolerance_optmz1 = tolerance_optmz1
        self.tolerance_optmz2 = tolerance_optmz2
        self.constraints: dict = constraints

        # ================================ CLASS OUTPUT ================================================================
        self.func_val_optimized = None
        self.pars_optimized = None
        self.omega = None

    def __check_func_h_pars(self):
        user_pars = self.func_h.__code__.co_varnames[:2]
        if len(user_pars) != 2:
            raise Exception("Your function must have exactly 2 parameters called 'pars_vect' and 'data'")
        if len({"pars_vect", "data"}.difference(user_pars)) != 0:
            raise Exception("Your function's parameters name must be 'pars_vect' and 'data'")

    def __check_missspecification(self):
        if self.k < max(self.initial_guess.shape):
            raise Exception("You've inserted a number of conditions less than the number of parameters to estimate. "
                            "You cannot estimate model parameter")

    def __calc_g_hat(self, pars_vect: np.array):
        """
        Quello che ritorna da func_h() può avere sulle righe la numerosità campionaria e sulle colonne le condizioni,
        oppure l'opposto. In entrambi i casi _calc_g_hat() fixa la cosa.
        """
        tmp_arr: np.array = self.func_h(pars_vect=pars_vect, data=self.data)
        # check if moment conditions are on the columns
        tmp_arr = tmp_arr if tmp_arr.shape[0] > tmp_arr.shape[1] else tmp_arr.transpose()
        return np.mean(tmp_arr, axis=0)  # shape: (num_conditions, ); vector 1D

    def _obj_func(self, pars_vect: np.array, omega: np.array):

        g_hat: np.array = self.__calc_g_hat(pars_vect=pars_vect)  # return 1D vector
        inv_omega = np.linalg.inv(omega)

        # given the 1D dimension of g_hat, the transpose is not necessary
        return g_hat @ inv_omega @ g_hat

    def _calc_gradient(self):

        if not self.func_val_optimized:
            self._optimizer_boss()

        h = 0.00001
        n = max(self.initial_guess.shape)
        pos = np.identity(n)
        # il gradiente avrà k righe (condizioni) e n colonne (parametri)
        grad = np.zeros((self.k, n))

        beta = self.pars_optimized

        for i in range(n):
            x: np.array = self.__calc_g_hat(np.multiply(beta, (1 + h * pos[:, i])))
            xx: np.array = self.__calc_g_hat(beta)

            # grad sarà una matrice diagonale con
            grad[:, i] = ((x - xx) / (beta[i] * h))

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

            print("\nPer problemi, si prega di contattare Kevyn Stefanelli +39 320 647 5439")

            print('\n=================================================================================================')
            print(f'\n{summary}')
            print('\n================================= ENJOY ECONOMETRICS GUYS =======================================')

        return summary

if __name__ == "__main__":

    from scipy.stats import t
    import random

    # ================= RANDOM SAMPLE BY T-DISTRIBUTION ================================================================
    np.random.seed(100)
    data = t.rvs(11, size=5_000_000)
    # data = np.random.normal(1, 1, 1000)

    # ================= DEFINE FUNCTION ================================================================================

    def func_moment_cond(pars_vect: np.array, data: np.array):
        """
        Norme buona scrittura:
         i) h: rappresentano un vettore che contengono l'n-esima condizione applicata ai dati input (i.e. data)
        ii) moment_conditions: è una matrice che ha su una dimensione la lunghezza campionaria e sull'altra il numero
        di condizioni applicate (ex. matrice(100x10) significa 100 dati campionari e 10 condizioni applicate)
        """
        h_1 = data ** 2 - pars_vect / (pars_vect - 2)
        h_2 = data ** 4 - (3 * pars_vect ** 2) / ((pars_vect - 2) * (pars_vect - 4))
        moment_conditions = np.vstack((h_1, h_2)).transpose()
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
        num_conditions=2,
        initial_guess=np.array([4.5]),
        GMM_lags=0,
        bounds=[[4.00001, None]],
        max_iter=2,
        tolerance_optmz1=0.00000000000001,
        tolerance_optmz2=0.00000000000001,
        constraints=None  # we don't have any constraints for this case
    )

    GMM.get_summary(printing=True)



