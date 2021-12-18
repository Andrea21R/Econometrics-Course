import pandas as pd
import numpy as np
from maxlik import MaxLik


def AR_1_log_like(pars_vect: np.array, data: np.array):
    T = data.shape[0]
    c = pars_vect[0]
    phi = pars_vect[1]
    sig2 = pars_vect[2]
    err = np.zeros((T,1))
    err[0] = data[0]-c
    for i in range(1,len(data)):
        err[i]=data[i]-c-phi*data[i-1]
    lik = np.zeros((T,1))
    for i in range(0,len(data)):
        lik[i] = -(1/2)*np.log(2*np.pi*sig2)-(err[i]**2)/(2*sig2)
    return lik

def MA_1_log_like(pars_vect: np.array, data: np.array):
    T = data.shape[0]
    c = pars_vect[0]
    psi = pars_vect[1]
    sig2 = pars_vect[2]
    err = np.zeros((T,1))
    err[0] = data[0]-c
    for i in range(1,len(data)):
        err[i] = data[i]-c-psi*err[i-1]
    lik = np.zeros((T,1))
    for i in range(0,len(data)):
        lik[i] = -(1/2)*np.log(2*np.pi*sig2)-(err[i]**2)/2*sig2
    return lik


if __name__ == "__main__":
    data: pd.DataFrame = pd.read_csv("~/PycharmProjects/Econometrics-Course/Dataset/GDP.csv", index_col="DATE")

    log_gdp: np.array = np.log(data).diff().dropna().to_numpy()
    initial_guess = np.array([0, 0.8, 0.1])
    bounds = [[-9999, 9999], [-0.9999, 0.9999], [0.0001, 9999]]

    for method in ["sandwich", "outer", None]:
        maxlik = MaxLik(
            func_vector=AR_1_log_like,
            data=log_gdp,
            initial_guess=initial_guess,
            bounds=bounds,
            method_se_optimization=method
        )
        maxlik.get_summary(printing=True)

        print("\n=========================================================================================\n")

        maxlik = MaxLik(
            func_vector=MA_1_log_like,
            data=log_gdp,
            initial_guess=initial_guess,
            bounds=bounds,
            method_se_optimization=method
        )
        maxlik.get_summary(printing=True)

