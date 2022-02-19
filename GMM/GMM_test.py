import numpy as np
from scipy.stats import t
import random

from GMM import GMM
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
    h_2 = data ** 4 - (3 * pars_vect ** 2) / \
        ((pars_vect - 2) * (pars_vect - 4))
    moment_conditions = np.vstack((h_1, h_2)).transpose()
    return moment_conditions

def func_g_hat(pars_vect, data):
    T = len(data)
    h_1 = (1 / T) * sum(data ** 2) - pars_vect / (pars_vect - 2)
    h_2 = (1 / T) * sum(data ** 4) - (3 * pars_vect ** 2) / \
        ((pars_vect - 2) * (pars_vect - 4))
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
