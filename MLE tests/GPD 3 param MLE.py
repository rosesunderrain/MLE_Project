import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy import stats as st

m = 100
means = np.empty(shape = m)
sigmas = np.empty(shape = m)
xis = np.empty(shape = m)

for i in range(m):
    n = 10000
    mu = 20000
    sigma = 150
    xi = 0.2
    probs = np.random.uniform(0, 1, n)
    def GPD_inv(p, mu, sigma, xi):
        return mu + sigma / xi * (np.power(p, -1 * xi) - 1)
    sample = GPD_inv(probs, mu, sigma, xi)

    # estimating the parameters with MLE method

    #def MLE_function(param):
    #    marker1 = (1 + param[2] / param[1] * (sample - param[0])) < 0
    #    if np.sum(marker1) > 0:
    #        return 1e30
    #    else:
    #        f_x = 1 / param[1] * (np.power(1 + param[2] / param[1] * (sample - param[0]), -1 / param[2] - 1))
    #        marker2 = f_x < 1e-20
    #        f_x[marker2] = 1e-20
    #        return -1 * np.sum(
    #            np.log(f_x)
    #        )

    def MLE_function(param):
        f_x = 1 / param[1] * (np.power(1 + param[2] / param[1] * (sample - param[0]), -1 / param[2] - 1))
        marker2 = f_x < 1e-20
        f_x[marker2] = 1e-20
        return -1 * np.sum(
            np.log(f_x)
        )

    mu_upperbound = np.min(sample)

    b0 = [0.0000001, mu_upperbound]
    b1 = [0.0000001, 10000]
    b2 = [0.0000001, 0.4999999]

    bnds = (b0, b1, b2)

    # starting point should be high enough

    param0 = (50000, 43, 0.111)

    solution = minimize(MLE_function, param0, bounds = bnds)
    means[i] = solution.x[0]
    sigmas[i] = solution.x[1]
    xis[i] = solution.x[2]

av_mean = np.average(means)
av_sigma = np.average(sigmas)
av_xi = np.average(xis)

print(av_mean)
print(av_sigma)
print(av_xi)