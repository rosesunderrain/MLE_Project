import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy import stats as st

# making the sample

n = 1000
mean = 7
sigma = 3
probs = np.random.uniform(0, 1, n)
sample = st.norm.ppf(probs, loc = mean, scale = sigma)

# estimating the parameters with MLE method

def MLE_function(param):
    return -1 * np.sum(np.log((st.norm.pdf(sample, loc = param[0], scale = param[1]))))

param0 = (15, 8)

b = [0.5, 10000]

bnds = (b, b)

solution = minimize(MLE_function, param0, bounds = bnds)

print(type(param0))