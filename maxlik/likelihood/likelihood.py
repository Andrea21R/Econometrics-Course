import numpy as np


def logLikeAR_1(parameters, data):
    c = parameters[0]
    phi = parameters[1]
    sigma_2 = parameters[2]
    Lik = np.zeros(len(data))
    Lik[0] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma_2 / (1 - phi ** 2)) - ((1 - phi ** 2) / 2 * sigma_2) * (
                data[0] - c / (1 - phi)) ** 2
    for i in range(1, len(data)):
        Lik[i] = -0.5 * np.log(2 * np.pi) - 0.5 * np.log(sigma_2) - 0.5 * (
                    data[i] - c - phi * data[i - 1]) ** 2 / sigma_2
    return Lik

