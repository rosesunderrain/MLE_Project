import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import fsolve
from scipy import stats as st

np.random.seed(123)

# importing the data

# data = pd.read_csv('/Users/marcellszegedi/Documents/P2A_wannabe.csv', sep = ";")
n = 10000
mu = 20000
sigma = 150
xi = 0.3
probs = np.random.uniform(0, 1, n)
def GPD_inv(p, mu, sigma, xi):
    return mu + sigma / xi * (np.power(p, -1 * xi) - 1)
data = GPD_inv(probs, mu, sigma, xi)

# setting the quantiles

p_1 = 0.25
p_2 = 0.75

# getting the quantiles of the input data

data_p_1 = np.quantile(data, p_1)
data_p_2 = np.quantile(data, p_2)

# solving the system of equation

def equations(variables):
    v_sigma = variables[0]
    v_xi = variables[1]
    eq_1 = mu + v_sigma / v_xi * (1 - np.power(1 - p_1, v_xi))-data_p_1
    eq_2 = mu + v_sigma / v_xi * (1 - np.power(1 - p_2, v_xi))-data_p_2
    return (eq_1, eq_2)

solution = fsolve(equations, (100, 0.1))
print((data_p_1, data_p_2))