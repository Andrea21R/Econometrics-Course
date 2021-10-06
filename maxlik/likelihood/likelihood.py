import numpy as np
from statsmodels.tsa.arima_process import ArmaProcess
from scipy.optimize import minimize


def ar1_loglike(parameters, data):

    c = parameters[0]
    phi = parameters[1]
    sigma_2 = parameters[2]
    lik = np.zeros(len(data))
    lik[0] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma_2 / (1 - phi ** 2)) - ((1 - phi ** 2) / 2 * sigma_2) * (
                data[0] - c / (1 - phi)) ** 2
    for i in range(1, len(data)):
        lik[i] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma_2) - 0.5 * (
                    data[i] - c - phi * data[i - 1]) ** 2 / sigma_2
    return lik


def garch_loglik(parameters, data):

    mu = parameters[0]
    omega = parameters[1]
    alpha = parameters[2]
    beta = parameters[3]

    sig2 = np.zeros(len(data))
    sig2[0] = (data[0] - mu) ** 2
    epsilon = data - mu

    for i in range(1, len(data)):
        sig2[i] = omega + (alpha * epsilon[i-1] ** 2) + (beta * sig2[i-1])
    lik = np.zeros(len(data))

    lik[0] = (-0.5 * np.log(2 * np.pi)) - 0.5 * epsilon[0] ** 2 / sig2[0]

    for i in range(1, len(data)):
        lik[i] = - 0.5 * np.log(2 * np.pi) - 0.5 * (np.log(sig2[i]) + (epsilon[i] ** 2 / sig2[i]))

    return lik
