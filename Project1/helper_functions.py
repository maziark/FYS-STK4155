import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def franke_function(x, y):
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-(9 * x - 7) ** 2 / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-(9 * x - 4) ** 2 - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def MSE(y, y_tilde):
    return np.mean(np.power(y - y_tilde, 2))


def R2(y, y_tilde):
    y_mean = np.mean(y)
    return 1 - (MSE(y, y_tilde)) / MSE(y, y_mean)


def var(y):
    return MSE(y, np.mean(y))


