import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy import stats as st

# making the sample

n = 10000
mu = 20000
sigma = 100
xi = 0.3
probs = np.random.uniform(0, 1, n)
def GPD_inv(p, mu, sigma, xi):
    return mu + sigma / xi * (np.power(p, -1 * xi) - 1)
sample = GPD_inv(probs, mu, sigma, xi)

# estimating the parameters with MLE method

def MLE_function(param):
    f_x = 1 / param[0] * (np.power(1 + param[1] / param[0] * (sample - mu), -1 / param[1] - 1))
    marker = f_x < 1e-20
    f_x[marker] = 1e-20
    return -1 * np.sum(
        np.log(f_x)
    )

b0 = [0.0000001, 10000]
b1 = [0.0000001, 0.4999999]

bnds = (b0, b1)

param0 = (43, 0.111)

solution = minimize(MLE_function, param0, bounds = bnds)
print(solution)

